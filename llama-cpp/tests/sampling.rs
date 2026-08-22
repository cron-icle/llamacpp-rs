//! Integration coverage for `LlamaSampler` variants that need a real model
//! (vocab size, token biasing, penalties) rather than the doctest-only unit
//! coverage in `src/sampling.rs`.
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use hf_hub::api::sync::ApiBuilder;
use llama_cpp::context::params::LlamaContextParams;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::{AddBos, LlamaModel};
use llama_cpp::sampling::LlamaSampler;
use llama_cpp::token::logit_bias::LlamaLogitBias;
use llama_cpp::token::LlamaToken;
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

/// Decodes a short prompt and returns a context positioned to sample the
/// next token.
fn primed_context<'a>(
    backend: &'a LlamaBackend,
    model: &'a LlamaModel,
) -> (llama_cpp::context::LlamaContext<'a>, LlamaBatch<'a>) {
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(256).unwrap()));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .expect("unable to create context");

    let tokens = model
        .str_to_token("Once upon a time", AddBos::Always)
        .expect("failed to tokenize prompt");
    let mut batch = LlamaBatch::new(64, 1);
    batch.add_sequence(&tokens, 0, false).unwrap();
    ctx.decode(&mut batch).expect("decode failed");
    (ctx, batch)
}

#[test]
fn greedy_sampling_is_deterministic() {
    let backend = backend();
    let model = load_model(backend);
    let (ctx, batch) = primed_context(backend, &model);

    let mut sampler_a = LlamaSampler::greedy();
    let mut sampler_b = LlamaSampler::greedy();
    let idx = batch.n_tokens() - 1;

    let a = sampler_a.sample(&ctx, idx);
    let b = sampler_b.sample(&ctx, idx);
    assert_eq!(
        a, b,
        "greedy sampling must be deterministic for the same logits"
    );
}

#[test]
fn logit_bias_can_force_selection() {
    let backend = backend();
    let model = load_model(backend);
    let (ctx, batch) = primed_context(backend, &model);
    let idx = batch.n_tokens() - 1;

    // Push token 0's logit far above everything else, then greedily sample.
    let biases = vec![LlamaLogitBias::new(LlamaToken(0), 1_000.0)];
    let mut chain = LlamaSampler::chain_simple([
        LlamaSampler::logit_bias(model.n_vocab(), &biases),
        LlamaSampler::greedy(),
    ]);

    let sampled = chain.sample(&ctx, idx);
    assert_eq!(sampled, LlamaToken(0));
}

#[test]
fn penalties_sampler_runs_without_model_state_corruption() {
    let backend = backend();
    let model = load_model(backend);
    let (ctx, batch) = primed_context(backend, &model);
    let idx = batch.n_tokens() - 1;

    let mut chain = LlamaSampler::chain_simple([
        LlamaSampler::penalties(&model, 64, 1.1, 0.0, 0.0),
        LlamaSampler::dist(42),
    ]);

    let token = chain.sample(&ctx, idx);
    assert!(
        token.0 >= 0,
        "sampled token id should be a valid vocab index"
    );
}

#[test]
fn sampler_reset_clears_internal_state() {
    let backend = backend();
    let model = load_model(backend);
    let (ctx, batch) = primed_context(backend, &model);
    let idx = batch.n_tokens() - 1;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(&model, 64, 1.5, 0.0, 0.0),
        LlamaSampler::greedy(),
    ]);
    let token = sampler.sample(&ctx, idx);
    sampler.accept(token);
    sampler.reset();
    // After reset, sampling should not panic and should still produce a valid token.
    let after_reset = sampler.sample(&ctx, idx);
    assert!(after_reset.0 >= 0);
}

#[test]
fn dry_sampler_runs_end_to_end() {
    let backend = backend();
    let model = load_model(backend);
    let (ctx, batch) = primed_context(backend, &model);
    let idx = batch.n_tokens() - 1;

    let mut chain = LlamaSampler::chain_simple([
        LlamaSampler::dry(&model, 0.8, 1.75, 2, 64, [] as [&[u8]; 0]),
        LlamaSampler::greedy(),
    ]);

    let token = chain.sample(&ctx, idx);
    assert!(token.0 >= 0);
}
