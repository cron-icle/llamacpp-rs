//! Hard-case tests for `LlamaModel::load_from_file`: correctness of metadata
//! read back from a real (tiny) GGUF model, and proper `Err` returns (never
//! a panic) for missing/corrupt/malformed inputs.
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use hf_hub::api::sync::ApiBuilder;
use llamacpp_rs::llama_backend::LlamaBackend;
use llamacpp_rs::model::params::LlamaModelParams;
use llamacpp_rs::model::{AddBos, LlamaModel};
use llamacpp_rs::token::LlamaToken;
use llamacpp_rs::LlamaModelLoadError;
use std::io::Write;
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

/// The llama backend can only be initialized once per process, so every test
/// in this binary must share the same instance.
fn backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| LlamaBackend::init().expect("unable to init llama backend"))
}

/// A scratch file under the OS temp dir, removed on drop. Not the `tempfile`
/// crate (not a dependency of this workspace) — just enough to avoid
/// colliding with parallel test runs.
struct ScratchFile(PathBuf);

impl ScratchFile {
    fn new(name: &str, contents: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "llamacpp-rs-test-{name}-{}-{:?}.gguf",
            std::process::id(),
            std::thread::current().id()
        ));
        let mut f = std::fs::File::create(&path).expect("failed to create scratch file");
        f.write_all(contents).expect("failed to write scratch file");
        Self(path)
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// A valid tiny model loads, and its metadata matches the values known for
/// `tinyllamas/stories260K.gguf` (verified independently by directly
/// querying the model and cross-checked against llama.cpp's own
/// `print_info` startup log for the same file) — not just "it loaded".
#[test]
fn valid_model_loads_with_correct_metadata() {
    let backend = backend();
    let model_params = pin!(LlamaModelParams::default());
    let model = LlamaModel::load_from_file(backend, tiny_model_path(), &model_params)
        .expect("valid tiny model should load successfully");

    assert_eq!(model.n_vocab(), 512, "vocab size should be 512");
    assert_eq!(model.n_ctx_train(), 2048, "training context should be 2048");
    assert_eq!(model.n_embd(), 64, "embedding width should be 64");
    assert_eq!(model.n_layer(), 5, "layer count should be 5");
    assert_eq!(model.n_head(), 8, "attention head count should be 8");
    assert_eq!(
        model.meta_val_str("general.architecture").as_deref(),
        Ok("llama"),
        "architecture metadata should be 'llama'"
    );

    // Tokenization output for a fixed prompt is also part of "loaded and
    // returning data properly": these exact ids are what the tokenizer
    // embedded in this specific GGUF produces for this exact prompt.
    let tokens = model
        .str_to_token("Once upon a time", AddBos::Always)
        .expect("tokenization should succeed on a loaded model");
    assert_eq!(
        tokens,
        vec![
            LlamaToken(1),
            LlamaToken(403),
            LlamaToken(407),
            LlamaToken(261),
            LlamaToken(378),
        ],
        "tokenizer output for a fixed prompt should be exact and reproducible"
    );
}

/// Loading from a path that does not exist must return a proper `Err`, never
/// panic (regression test: `load_from_file` used to `debug_assert!` that the
/// path exists, which panicked instead of returning `Err` in debug/test
/// builds).
#[test]
fn missing_file_returns_err_not_panic() {
    let backend = backend();
    let model_params = pin!(LlamaModelParams::default());
    let missing = std::env::temp_dir().join("llamacpp-rs-this-file-does-not-exist-1234.gguf");
    assert!(
        !missing.exists(),
        "precondition: scratch path must not exist"
    );

    let result = LlamaModel::load_from_file(backend, &missing, &model_params);
    assert!(
        matches!(result, Err(LlamaModelLoadError::NullResult)),
        "expected Err(NullResult) for a missing file, got {result:?}"
    );
}

/// A file that looks like it could be a GGUF (right extension) but contains
/// arbitrary junk bytes must return `Err`, not panic or hang.
#[test]
fn junk_bytes_file_returns_err_not_panic() {
    let backend = backend();
    let model_params = pin!(LlamaModelParams::default());
    let scratch = ScratchFile::new("junk", &[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03]);

    let result = LlamaModel::load_from_file(backend, &scratch.0, &model_params);
    assert!(
        matches!(result, Err(LlamaModelLoadError::NullResult)),
        "expected Err(NullResult) for a junk-bytes file, got {result:?}"
    );
}

/// A plain text file renamed to a `.gguf` extension (missing the GGUF magic
/// entirely) must return `Err`, not panic or hang.
#[test]
fn text_file_renamed_gguf_returns_err_not_panic() {
    let backend = backend();
    let model_params = pin!(LlamaModelParams::default());
    let scratch = ScratchFile::new(
        "text",
        b"this is definitely not a gguf file, just plain text\n",
    );

    let result = LlamaModel::load_from_file(backend, &scratch.0, &model_params);
    assert!(
        matches!(result, Err(LlamaModelLoadError::NullResult)),
        "expected Err(NullResult) for a text file, got {result:?}"
    );
}

/// A GGUF file truncated right after its magic/version header (valid magic,
/// no actual content) must return `Err`, not panic or hang.
#[test]
fn truncated_gguf_header_returns_err_not_panic() {
    let backend = backend();
    let model_params = pin!(LlamaModelParams::default());
    // "GGUF" magic + version 3, then nothing else - a real but empty/broken header.
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    let scratch = ScratchFile::new("truncated", &bytes);

    let result = LlamaModel::load_from_file(backend, &scratch.0, &model_params);
    assert!(
        matches!(result, Err(LlamaModelLoadError::NullResult)),
        "expected Err(NullResult) for a truncated GGUF header, got {result:?}"
    );
}

/// An empty path string must return `Err`, not panic.
#[test]
fn empty_path_returns_err_not_panic() {
    let backend = backend();
    let model_params = pin!(LlamaModelParams::default());

    let result = LlamaModel::load_from_file(backend, "", &model_params);
    assert!(
        matches!(result, Err(LlamaModelLoadError::NullResult)),
        "expected Err(NullResult) for an empty path, got {result:?}"
    );
}

/// A path containing a NUL byte cannot be turned into a C string and must
/// return `Err`, not panic.
#[test]
fn path_with_interior_nul_returns_err_not_panic() {
    let backend = backend();
    let model_params = pin!(LlamaModelParams::default());

    let result = LlamaModel::load_from_file(backend, "bad\0path.gguf", &model_params);
    assert!(
        matches!(result, Err(LlamaModelLoadError::NullError(_))),
        "expected Err(NullError) for a path with an interior NUL, got {result:?}"
    );
}

/// On Windows, a path built from an unpaired UTF-16 surrogate is valid as an
/// `OsString`/`Path` but cannot be converted to UTF-8 `str`; this must return
/// `Err(PathToStrError)`, not panic.
#[cfg(windows)]
#[test]
fn invalid_utf8_path_returns_err_not_panic() {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;

    let backend = backend();
    let model_params = pin!(LlamaModelParams::default());

    // 0xD800 is a lone high surrogate: valid as a Windows OsString component,
    // invalid as UTF-8/UTF-16 text.
    let invalid = OsString::from_wide(&[0xD800]);
    let path = PathBuf::from(invalid);
    assert!(
        path.to_str().is_none(),
        "precondition: constructed path must not be valid UTF-8"
    );

    let result = LlamaModel::load_from_file(backend, &path, &model_params);
    assert!(
        matches!(result, Err(LlamaModelLoadError::PathToStrError(_))),
        "expected Err(PathToStrError) for an invalid-UTF-8 path, got {result:?}"
    );
}
