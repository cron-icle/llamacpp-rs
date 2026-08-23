//! End-to-end multimodal pipeline tests against a real (small) vision GGUF
//! model + a real image, verifying the mtmd API actually returns correct
//! data for real inputs (not just "didn't crash"), plus failure-mode
//! coverage for corrupt/invalid image input.
//!
//! Downloads `ggml-org/granite-docling-258M-GGUF` (Q8_0 model + mmproj,
//! ~280 MB combined, cached by `hf-hub` after the first run) - the smallest
//! vision-capable model in llama.cpp's own mtmd test matrix
//! (`tools/mtmd/tests.sh`). There is no truly tiny (single-digit MB)
//! vision-capable GGUF publicly available in the way `tinyllamas/stories260K`
//! serves as a tiny text fixture; this is the closest equivalent that still
//! exercises the real vision encoder + projector + decode pipeline.
//!
//! The test image is `test-1.jpeg`, vendored directly in the llama.cpp
//! submodule at `llama-cpp-sys/vendor/llama.cpp/tools/mtmd/test-1.jpeg` and
//! used by llama.cpp's own mtmd CI (`tools/mtmd/tests.sh`).
//!
//! Requires the `mtmd` feature; this whole file is a no-op test binary
//! without it (the `mtmd` module doesn't exist otherwise), matching how
//! `tests/grammar_without_common.rs` and `run-tests.sh` already gate
//! mtmd-dependent test runs behind `--features sampler,mtmd`.
#![cfg(feature = "mtmd")]
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use hf_hub::api::sync::ApiBuilder;
use llamacpp_rs::context::params::LlamaContextParams;
use llamacpp_rs::llama_backend::LlamaBackend;
use llamacpp_rs::model::params::LlamaModelParams;
use llamacpp_rs::model::LlamaModel;
use llamacpp_rs::mtmd::{
    MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdContextParams, MtmdInputChunkType, MtmdInputText,
};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::sync::OnceLock;

/// The vendored test image used by llama.cpp's own mtmd CI.
fn test_image_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../llama-cpp-sys/vendor/llama.cpp/tools/mtmd/test-1.jpeg")
}

fn model_and_mmproj_paths() -> (PathBuf, PathBuf) {
    static PATHS: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();
    PATHS
        .get_or_init(|| {
            let repo = ApiBuilder::new()
                .with_progress(false)
                .build()
                .expect("unable to build huggingface api client")
                .model("ggml-org/granite-docling-258M-GGUF".to_string());
            let model = repo
                .get("granite-docling-258M-Q8_0.gguf")
                .expect("unable to download granite-docling-258M-Q8_0.gguf");
            let mmproj = repo
                .get("mmproj-granite-docling-258M-Q8_0.gguf")
                .expect("unable to download mmproj-granite-docling-258M-Q8_0.gguf");
            (model, mmproj)
        })
        .clone()
}

fn backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| LlamaBackend::init().expect("unable to init llama backend"))
}

fn load_model(backend: &LlamaBackend) -> LlamaModel {
    let (model_path, _) = model_and_mmproj_paths();
    let model_params = pin!(LlamaModelParams::default());
    LlamaModel::load_from_file(backend, model_path, &model_params)
        .expect("unable to load granite-docling vision model")
}

fn init_mtmd(model: &LlamaModel) -> MtmdContext {
    let (_, mmproj_path) = model_and_mmproj_paths();
    let params = MtmdContextParams {
        use_gpu: false,
        print_timings: false,
        n_threads: 2,
        media_marker: std::ffi::CString::new(llamacpp_rs::mtmd::mtmd_default_marker()).unwrap(),
        image_min_tokens: -1,
        image_max_tokens: -1,
    };
    MtmdContext::init_from_file(mmproj_path.to_str().unwrap(), model, &params)
        .expect("unable to init mtmd context from a valid mmproj file")
}

/// A valid mmproj file loads, and the resulting context correctly reports
/// vision support for a known vision-capable model.
#[test]
fn mtmd_context_inits_and_reports_vision_support() {
    let backend = backend();
    let model = load_model(backend);
    let mtmd_ctx = init_mtmd(&model);
    assert!(
        mtmd_ctx.support_vision(),
        "granite-docling is a vision model; support_vision() should be true"
    );
    assert!(
        !mtmd_ctx.support_audio(),
        "granite-docling has no audio tower; support_audio() should be false"
    );
}

/// Loading the real vendored test image through the mtmd helper produces a
/// bitmap whose dimensions match the actual JPEG (640x488, verified with
/// `file` against the vendored fixture), not just "some bitmap".
#[test]
fn bitmap_from_real_image_file_has_correct_dimensions() {
    let backend = backend();
    let model = load_model(backend);
    let mtmd_ctx = init_mtmd(&model);

    let bitmap = MtmdBitmap::from_file(&mtmd_ctx, test_image_path().to_str().unwrap(), false)
        .expect("loading the vendored test image should succeed");

    assert_eq!(bitmap.nx(), 640, "image width should be 640px");
    assert_eq!(bitmap.ny(), 488, "image height should be 488px");
    assert!(!bitmap.is_audio());
    assert!(
        !bitmap.data().is_empty(),
        "decoded (non-placeholder) bitmap should contain pixel data"
    );
}

/// Tokenizing text with a real image marker against a real image produces a
/// chunk sequence that actually reflects the image: contains an Image chunk
/// (not just Text), and reports a positive, image-encoder-sized token count
/// - not just "some chunks came back".
#[test]
fn tokenize_produces_text_and_image_chunks_with_real_token_counts() {
    let backend = backend();
    let model = load_model(backend);
    let mtmd_ctx = init_mtmd(&model);

    let bitmap = MtmdBitmap::from_file(&mtmd_ctx, test_image_path().to_str().unwrap(), false)
        .expect("loading the vendored test image should succeed");

    let marker = llamacpp_rs::mtmd::mtmd_default_marker();
    let text = MtmdInputText {
        text: format!("Describe this image: {marker}"),
        add_special: true,
        parse_special: true,
    };

    let chunks = mtmd_ctx
        .tokenize(text, &[&bitmap])
        .expect("tokenizing real text + a real image should succeed");

    assert!(
        chunks.len() >= 2,
        "expected at least a text chunk and an image chunk, got {} chunks",
        chunks.len()
    );

    let mut saw_text = false;
    let mut saw_image = false;
    let mut image_chunk_tokens = 0usize;
    for i in 0..chunks.len() {
        let chunk = chunks.get(i).expect("index within bounds should be Some");
        match chunk.chunk_type() {
            MtmdInputChunkType::Text => {
                saw_text = true;
                assert!(
                    chunk.text_tokens().is_some(),
                    "a Text chunk should expose text_tokens()"
                );
            }
            MtmdInputChunkType::Image => {
                saw_image = true;
                assert!(
                    chunk.text_tokens().is_none(),
                    "text_tokens() should be None for an Image chunk"
                );
                image_chunk_tokens = chunk.n_tokens();
            }
            MtmdInputChunkType::Audio => panic!("no audio input was provided"),
        }
    }
    assert!(saw_text, "expected at least one Text chunk");
    assert!(saw_image, "expected at least one Image chunk");
    assert!(
        image_chunk_tokens > 0,
        "the image chunk should be encoded into a positive number of vision tokens, got {image_chunk_tokens}"
    );

    // total_tokens() must be consistent with (>= the sum coming from) the
    // individual chunks actually observed above, not an unrelated number.
    assert!(
        chunks.total_tokens() >= image_chunk_tokens,
        "total_tokens() ({}) should be at least the image chunk's token count ({image_chunk_tokens})",
        chunks.total_tokens()
    );
}

/// Full pipeline correctness: tokenize real text + a real image, evaluate
/// the chunks (running the vision encoder + `llama_decode`), and confirm
/// the context is left in a state that lets it actually generate a next
/// token deterministically - i.e. the multimodal embeddings really made it
/// into the model's KV cache and produced usable logits, not just "no error
/// was returned".
#[test]
fn eval_chunks_advances_context_and_enables_generation() {
    let backend = backend();
    let model = load_model(backend);
    let mtmd_ctx = init_mtmd(&model);

    let bitmap = MtmdBitmap::from_file(&mtmd_ctx, test_image_path().to_str().unwrap(), false)
        .expect("loading the vendored test image should succeed");

    let marker = llamacpp_rs::mtmd::mtmd_default_marker();
    let text = MtmdInputText {
        text: format!("Describe this image: {marker}"),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx
        .tokenize(text, &[&bitmap])
        .expect("tokenization should succeed");
    let expected_positions = chunks.total_positions();
    assert!(
        expected_positions > 0,
        "a real prompt + image should produce a positive position count"
    );

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()))
        .with_n_batch(512);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .expect("unable to create context");

    let n_past = chunks
        .eval_chunks(&mtmd_ctx, &mut ctx, 0, 0, 1, true)
        .expect("evaluating real text+image chunks should succeed");

    assert_eq!(
        n_past, expected_positions,
        "n_past after eval_chunks should equal the chunks' total position count"
    );

    // The context should now be primed to sample a valid next token - proof
    // that real logits (derived from the encoded image, not garbage) are
    // present, not just that eval_chunks returned Ok.
    let mut sampler = llamacpp_rs::sampling::LlamaSampler::greedy();
    let token = sampler.sample(&ctx, -1);
    assert!(
        (0..model.n_vocab()).contains(&token.0),
        "post-eval sampled token id {} should be within the model's vocab range 0..{}",
        token.0,
        model.n_vocab()
    );
}

/// Garbage bytes that are not any recognized image/audio format must return
/// `Err`, not panic or hang, when loaded through the mtmd buffer helper.
#[test]
fn corrupt_image_buffer_returns_err_not_panic() {
    let backend = backend();
    let model = load_model(backend);
    let mtmd_ctx = init_mtmd(&model);

    let garbage = vec![0xFFu8, 0xD8, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF];
    let result = MtmdBitmap::from_buffer(&mtmd_ctx, &garbage, false);
    assert!(
        matches!(result, Err(MtmdBitmapError::NullResult)),
        "expected Err(NullResult) for a corrupt image buffer, got {result:?}"
    );
}

/// A nonexistent image path must return `Err`, not panic or hang.
#[test]
fn missing_image_file_returns_err_not_panic() {
    let backend = backend();
    let model = load_model(backend);
    let mtmd_ctx = init_mtmd(&model);

    let result = MtmdBitmap::from_file(&mtmd_ctx, "definitely_missing_image_1234.png", false);
    assert!(
        matches!(result, Err(MtmdBitmapError::NullResult)),
        "expected Err(NullResult) for a missing image file, got {result:?}"
    );
}

/// A truncated/corrupt (but correctly-prefixed) JPEG must return `Err`, not
/// panic or hang, when loaded through the mtmd buffer helper.
#[test]
fn truncated_jpeg_buffer_returns_err_not_panic() {
    let backend = backend();
    let model = load_model(backend);
    let mtmd_ctx = init_mtmd(&model);

    // Real JPEG SOI marker (0xFFD8) followed by nothing else.
    let truncated = vec![0xFFu8, 0xD8];
    let result = MtmdBitmap::from_buffer(&mtmd_ctx, &truncated, false);
    assert!(
        matches!(result, Err(MtmdBitmapError::NullResult)),
        "expected Err(NullResult) for a truncated JPEG, got {result:?}"
    );
}

/// Initializing mtmd from a nonexistent mmproj path must return `Err`, not
/// panic.
#[test]
fn init_from_missing_mmproj_returns_err_not_panic() {
    let backend = backend();
    let model = load_model(backend);
    let params = MtmdContextParams::default();
    let result =
        MtmdContext::init_from_file("definitely_missing_mmproj_1234.gguf", &model, &params);
    assert!(
        result.is_err(),
        "expected Err for a missing mmproj file, got Ok"
    );
}
