//! End-to-end embeddings tests against a real (tiny) GGUF model.
//!
//! Embeddings are one of the crate's headline user-facing features
//! (alongside chat completion and multimodal input), exercised here via
//! [`llamacpp_rs::context::LlamaContext::embeddings_seq_ith`] with
//! mean pooling enabled, mirroring `examples/embeddings`.
//!
//! Uses the same `tinyllamas/stories260K.gguf` model as the inference tests
//! (~1.1 MB, cached by `hf-hub` after the first run). Requires network access
//! on first run.
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use hf_hub::api::sync::ApiBuilder;
use llamacpp_rs::context::params::{LlamaContextParams, LlamaPoolingType};
use llamacpp_rs::llama_backend::LlamaBackend;
use llamacpp_rs::llama_batch::LlamaBatch;
use llamacpp_rs::model::params::LlamaModelParams;
use llamacpp_rs::model::{AddBos, LlamaModel};
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

fn embed(model: &LlamaModel, backend: &LlamaBackend, prompt: &str) -> Vec<f32> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(256).unwrap()))
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Mean);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .expect("unable to create embeddings context");

    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .expect("failed to tokenize prompt");

    let mut batch = LlamaBatch::new(64, 1);
    let last_index = (tokens.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
        batch
            .add(token, i, &[0], true)
            .expect("failed to add token to batch");
    }
    let _ = last_index;
    ctx.clear_kv_cache();
    ctx.decode(&mut batch).expect("llama_decode failed");

    ctx.embeddings_seq_ith(0)
        .expect("failed to read pooled embeddings")
        .to_vec()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

/// A context created with mean pooling and embeddings enabled must produce a
/// non-empty, non-degenerate pooled embedding vector for a decoded prompt.
#[test]
fn produces_nonzero_pooled_embedding() {
    let backend = backend();
    let model = load_model(backend);

    let embedding = embed(&model, backend, "Once upon a time");

    assert!(
        !embedding.is_empty(),
        "expected a non-empty embedding vector"
    );
    assert!(
        embedding.iter().any(|&v| v != 0.0),
        "expected at least one non-zero embedding component"
    );
}

/// Embeddings for a fixed prompt/model/pooling combination are deterministic
/// (no sampling involved), and the same prompt run twice must produce
/// bit-identical output.
#[test]
fn same_prompt_yields_identical_embedding() {
    let backend = backend();
    let model = load_model(backend);

    let first = embed(&model, backend, "The quick brown fox");
    let second = embed(&model, backend, "The quick brown fox");

    assert_eq!(
        first, second,
        "embeddings for the same prompt/model/pooling must be reproducible"
    );
}

/// Embeddings should be a meaningful semantic signal, not noise: two
/// unrelated prompts must be less cosine-similar to each other than a prompt
/// is to itself (similarity 1.0).
#[test]
fn distinct_prompts_yield_distinct_embeddings() {
    let backend = backend();
    let model = load_model(backend);

    let a = embed(&model, backend, "Once upon a time");
    let b = embed(&model, backend, "Quantum mechanics and general relativity");

    assert_eq!(
        a.len(),
        b.len(),
        "embeddings should share the same pooled width"
    );
    let similarity = cosine_similarity(&a, &b);
    assert!(
        similarity < 0.9999,
        "expected distinct prompts to yield distinct embeddings, got similarity {similarity}"
    );
}
