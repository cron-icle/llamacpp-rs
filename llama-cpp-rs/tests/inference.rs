//! End-to-end inference smoke tests against a real (tiny) GGUF model.
//!
//! These tests download `tinyllamas/stories260K.gguf` from the
//! `ggml-org/models` Hugging Face repo (~1.1 MB, cached by `hf-hub` after
//! the first run) and exercise the safe API against it: model loading,
//! tokenization, context/batch decoding, sampling, and KV-cache
//! manipulation. This is the same tiny model llama.cpp's own CI uses for
//! smoke-testing inference.
//!
//! Requires network access on first run. If the download fails (e.g. no
//! network), the test fails with a clear message rather than silently
//! skipping, since these are the only tests in the crate that exercise a
//! real inference path end-to-end.
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use encoding_rs::UTF_8;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_rs::context::params::LlamaContextParams;
use llama_cpp_rs::llama_backend::LlamaBackend;
use llama_cpp_rs::llama_batch::LlamaBatch;
use llama_cpp_rs::model::params::LlamaModelParams;
use llama_cpp_rs::model::{AddBos, LlamaModel};
use llama_cpp_rs::sampling::LlamaSampler;
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

/// The llama backend can only be initialized once per process (enforced by
/// `LlamaBackend::init`), so every test in this binary must share the same
/// instance rather than each calling `init()` independently.
fn backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| LlamaBackend::init().expect("unable to init llama backend"))
}

fn load_model(backend: &LlamaBackend) -> LlamaModel {
    let model_params = pin!(LlamaModelParams::default());
    LlamaModel::load_from_file(backend, tiny_model_path(), &model_params)
        .expect("unable to load tiny test model")
}

/// Loads the tiny model, tokenizes a prompt, decodes it, and greedily
/// samples a few tokens, exercising model/context/batch/sampling together.
#[test]
fn generates_tokens_from_prompt() {
    let backend = backend();
    let model = load_model(backend);

    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(256).unwrap()));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .expect("unable to create context");

    let prompt = "Once upon a time";
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .expect("failed to tokenize prompt");
    assert!(!tokens.is_empty(), "tokenizer produced no tokens");

    let mut batch = LlamaBatch::new(64, 1);
    let last_index = (tokens.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
        batch
            .add(token, i, &[0], i == last_index)
            .expect("failed to add token to batch");
    }
    ctx.decode(&mut batch).expect("llama_decode failed");

    let mut sampler =
        LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);
    let mut decoder = UTF_8.new_decoder();
    let mut n_cur = batch.n_tokens();
    let mut generated = String::new();

    for _ in 0..16 {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        generated.push_str(
            &model
                .token_to_piece(token, &mut decoder, true, None)
                .expect("failed to detokenize sampled token"),
        );

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .expect("failed to add sampled token to batch");
        n_cur += 1;
        ctx.decode(&mut batch).expect("llama_decode failed");
    }

    assert!(
        !generated.is_empty(),
        "expected at least one generated token from the tiny model"
    );
}

/// Clearing the KV cache mid-generation must not error and must allow the
/// context to accept a fresh sequence starting at position 0 again.
#[test]
fn kv_cache_clear_allows_reuse() {
    let backend = backend();
    let model = load_model(backend);

    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(256).unwrap()));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .expect("unable to create context");

    let tokens = model
        .str_to_token("Once upon a time", AddBos::Always)
        .expect("failed to tokenize prompt");

    let mut batch = LlamaBatch::new(64, 1);
    let last_index = (tokens.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
        batch
            .add(token, i, &[0], i == last_index)
            .expect("failed to add token to batch");
    }
    ctx.decode(&mut batch).expect("first decode failed");

    ctx.clear_kv_cache();

    batch.clear();
    for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
        batch
            .add(token, i, &[0], i == last_index)
            .expect("failed to add token to batch after kv cache clear");
    }
    ctx.decode(&mut batch)
        .expect("decode after kv cache clear should succeed");
}
