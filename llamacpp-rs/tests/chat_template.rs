//! End-to-end chat-templating tests against a real (tiny) GGUF model.
//!
//! Chat templating is the user-facing entry point for chat completion: a
//! caller supplies a list of `(role, content)` messages and gets back the
//! exact prompt string the model expects, ready to tokenize and decode. This
//! exercises [`llamacpp_rs::model::LlamaModel::apply_chat_template`] end to
//! end, independent of the raw-prompt inference path already covered by
//! `tests/inference.rs`.
//!
//! Uses the same `tinyllamas/stories260K.gguf` model as the inference tests
//! (~1.1 MB, cached by `hf-hub` after the first run). Requires network access
//! on first run.

use hf_hub::api::sync::ApiBuilder;
use llamacpp_rs::llama_backend::LlamaBackend;
use llamacpp_rs::model::params::LlamaModelParams;
use llamacpp_rs::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
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

fn messages() -> Vec<LlamaChatMessage> {
    vec![
        LlamaChatMessage::new(
            "system".to_string(),
            "You are a helpful assistant.".to_string(),
        )
        .expect("valid chat message"),
        LlamaChatMessage::new("user".to_string(), "Hello there".to_string())
            .expect("valid chat message"),
    ]
}

/// Applying an explicit "chatml" template renders the well-known `ChatML`
/// tag structure around each message, and (when `add_ass` is set) leaves the
/// prompt ending in an opening assistant tag ready for generation.
#[test]
fn chatml_template_renders_expected_structure() {
    let backend = backend();
    let model = load_model(backend);

    let tmpl = LlamaChatTemplate::new("chatml").expect("valid template name");
    let rendered = model
        .apply_chat_template(&tmpl, &messages(), true)
        .expect("failed to apply chat template");

    assert_eq!(
        rendered,
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
<|im_start|>user\nHello there<|im_end|>\n\
<|im_start|>assistant\n"
    );
}

/// With `add_ass = false`, the rendered prompt must not have a trailing
/// assistant tag appended, unlike the `add_ass = true` case above.
#[test]
fn add_ass_false_omits_trailing_assistant_tag() {
    let backend = backend();
    let model = load_model(backend);

    let tmpl = LlamaChatTemplate::new("chatml").expect("valid template name");
    let rendered = model
        .apply_chat_template(&tmpl, &messages(), false)
        .expect("failed to apply chat template");

    assert!(
        !rendered.ends_with("<|im_start|>assistant\n"),
        "expected no trailing assistant tag, got: {rendered:?}"
    );
    assert!(rendered.ends_with("<|im_end|>\n"));
}

/// `LlamaModel::chat_template` retrieves whatever template metadata (if any)
/// is baked into the GGUF file itself; the tiny stories model has none, so
/// this should surface as a clean error rather than a panic or garbage data.
#[test]
fn chat_template_missing_from_model_metadata_errors_cleanly() {
    let backend = backend();
    let model = load_model(backend);

    let result = model.chat_template(None);
    assert!(
        result.is_err(),
        "tiny stories model has no baked-in chat template metadata; expected an error, got {result:?}"
    );
}
