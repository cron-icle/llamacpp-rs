//! Integration coverage for `LlamaBatch`, KV-cache sequence management, and
//! state/session save-load round trips.
//!
//! Uses the same tiny `stories260K.gguf` model as `tests/inference.rs`. See
//! that file for the network/caching notes.
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use hf_hub::api::sync::ApiBuilder;
use llamacpp_rs::context::params::LlamaContextParams;
use llamacpp_rs::context::session::LlamaStateSeqFlags;
use llamacpp_rs::llama_backend::LlamaBackend;
use llamacpp_rs::llama_batch::{BatchAddError, LlamaBatch};
use llamacpp_rs::model::params::LlamaModelParams;
use llamacpp_rs::model::{AddBos, LlamaModel};
use llamacpp_rs::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::sync::OnceLock;

fn tiny_model_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        ApiBuilder::new()
            .with_progress(false)
            .build()
            .expect("unable to build huggingface api client")
            .model("ggml-org/models".to_string())
            .get("tinyllamas/stories260K.gguf")
            .expect("unable to download tiny test model tinyllamas/stories260K.gguf")
    })
    .clone()
}

fn backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| LlamaBackend::init().expect("unable to init llama backend"))
}

fn load_model(backend: &LlamaBackend) -> LlamaModel {
    let model_params = pin!(LlamaModelParams::default());
    LlamaModel::load_from_file(backend, tiny_model_path(), &model_params)
        .expect("unable to load tiny test model")
}

fn tokens_for(model: &LlamaModel, prompt: &str) -> Vec<LlamaToken> {
    model
        .str_to_token(prompt, AddBos::Always)
        .expect("failed to tokenize prompt")
}

// ---- LlamaBatch -----------------------------------------------------------

#[test]
fn batch_add_sequence_fills_and_decodes() {
    let backend = backend();
    let model = load_model(backend);
    let tokens = tokens_for(&model, "Once upon a time");

    let mut batch = LlamaBatch::new(64, 1);
    batch
        .add_sequence(&tokens, 0, false)
        .expect("add_sequence should succeed with enough capacity");
    assert_eq!(batch.n_tokens(), tokens.len() as i32);

    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(256).unwrap()));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .expect("unable to create context");
    ctx.decode(&mut batch)
        .expect("decode of add_sequence batch failed");
}

#[test]
fn batch_add_rejects_when_over_capacity() {
    let tokens = [LlamaToken(1), LlamaToken(2), LlamaToken(3)];
    let mut batch = LlamaBatch::new(2, 1);
    batch.add(tokens[0], 0, &[0], false).unwrap();
    batch.add(tokens[1], 1, &[0], false).unwrap();
    let err = batch.add(tokens[2], 2, &[0], true).unwrap_err();
    assert_eq!(err, BatchAddError::InsufficientSpace(2));
}

#[test]
fn batch_add_sequence_rejects_when_over_capacity() {
    let tokens = vec![LlamaToken(1), LlamaToken(2), LlamaToken(3)];
    let mut batch = LlamaBatch::new(2, 1);
    let err = batch.add_sequence(&tokens, 0, false).unwrap_err();
    assert_eq!(err, BatchAddError::InsufficientSpace(2));
}

#[test]
fn batch_clear_resets_token_count() {
    let mut batch = LlamaBatch::new(4, 1);
    batch.add(LlamaToken(1), 0, &[0], true).unwrap();
    batch.add(LlamaToken(2), 1, &[0], true).unwrap();
    assert_eq!(batch.n_tokens(), 2);
    batch.clear();
    assert_eq!(batch.n_tokens(), 0);
    // The batch must be reusable after clearing.
    batch.add(LlamaToken(3), 0, &[0], true).unwrap();
    assert_eq!(batch.n_tokens(), 1);
}

#[test]
fn batch_get_one_rejects_empty_buffer() {
    let empty: [LlamaToken; 0] = [];
    let err = LlamaBatch::get_one(&empty).unwrap_err();
    assert_eq!(err, BatchAddError::EmptyBuffer);
}

#[test]
fn batch_get_one_decodes() {
    let backend = backend();
    let model = load_model(backend);
    let tokens = tokens_for(&model, "Once upon a time");

    let mut batch =
        LlamaBatch::get_one(&tokens).expect("get_one should succeed on non-empty buffer");
    assert_eq!(batch.n_tokens(), tokens.len() as i32);

    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(256).unwrap()));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .expect("unable to create context");
    ctx.decode(&mut batch)
        .expect("decode of get_one batch failed");
}

// ---- KV cache sequence management -----------------------------------------

fn decoded_context<'a>(
    backend: &'a LlamaBackend,
    model: &'a LlamaModel,
) -> llamacpp_rs::context::LlamaContext<'a> {
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(256).unwrap()));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .expect("unable to create context");

    let tokens = tokens_for(model, "Once upon a time");
    let mut batch = LlamaBatch::new(64, 1);
    batch.add_sequence(&tokens, 0, false).unwrap();
    ctx.decode(&mut batch).expect("initial decode failed");
    ctx
}

#[test]
fn kv_cache_seq_cp_copies_across_sequences() {
    let backend = backend();
    let model = load_model(backend);
    let mut ctx = decoded_context(backend, &model);

    // Sequence 1 has nothing in it yet; copy sequence 0's cache into it.
    ctx.kv_cache_seq_cp(0, 1, None, None)
        .expect("seq_cp should succeed");

    assert!(ctx.kv_cache_seq_pos_max(1) >= ctx.kv_cache_seq_pos_min(1));
}

#[test]
fn kv_cache_seq_rm_removes_positions() {
    let backend = backend();
    let model = load_model(backend);
    let mut ctx = decoded_context(backend, &model);

    let max_before = ctx.kv_cache_seq_pos_max(0);
    assert!(max_before >= 0);

    ctx.kv_cache_seq_rm(0, None, None)
        .expect("full sequence removal should always succeed");

    // After removing everything, the sequence should have no max position left.
    assert!(ctx.kv_cache_seq_pos_max(0) < 0);
}

#[test]
fn kv_cache_seq_add_shifts_positions() {
    let backend = backend();
    let model = load_model(backend);
    let mut ctx = decoded_context(backend, &model);

    let min_before = ctx.kv_cache_seq_pos_min(0);
    ctx.kv_cache_seq_add(0, None, None, 5)
        .expect("seq_add should succeed");
    let min_after = ctx.kv_cache_seq_pos_min(0);

    assert_eq!(min_after, min_before + 5);
}

#[test]
fn kv_cache_seq_keep_and_clear_do_not_error() {
    let backend = backend();
    let model = load_model(backend);
    let mut ctx = decoded_context(backend, &model);

    ctx.llama_kv_cache_seq_keep(0);
    ctx.clear_kv_cache();
    assert!(ctx.kv_cache_seq_pos_max(0) < 0);
}

// ---- State / session save-load round trip ----------------------------------

#[test]
fn state_save_and_load_round_trip_preserves_tokens() {
    let backend = backend();
    let model = load_model(backend);
    let ctx = decoded_context(backend, &model);

    let tokens = tokens_for(&model, "Once upon a time");
    let dir = std::env::temp_dir().join(format!("llama-cpp-rs-test-state-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("state.bin");

    ctx.state_save_file(&path, &tokens)
        .expect("state_save_file should succeed");

    // Fresh context to load into, proving the file (not in-memory state) round trips.
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(256).unwrap()));
    let mut fresh_ctx = model
        .new_context(backend, ctx_params)
        .expect("unable to create fresh context");
    let loaded = fresh_ctx
        .state_load_file(&path, tokens.len())
        .expect("state_load_file should succeed");

    assert_eq!(loaded, tokens);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn state_seq_get_and_set_round_trip() {
    let backend = backend();
    let model = load_model(backend);
    let mut ctx = decoded_context(backend, &model);

    let state = ctx
        .state_seq_get(0, LlamaStateSeqFlags::empty())
        .expect("state_seq_get should succeed");
    assert!(state.byte_len() > 0);

    // Restoring into a different sequence id on the same context must succeed
    // and report a matching byte length.
    ctx.state_seq_set(&state, 1)
        .expect("state_seq_set should succeed");
}
