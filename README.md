# llamacpp-rs

Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp): a single
`llamacpp-rs` crate providing a safe wrapper close to the underlying C API,
with multimodal (`mtmd`) support enabled by default so text, vision, and
audio inputs all work out of the box.

This project tracks llama.cpp closely and does not follow semver
meaningfully — pin an exact version if you need stability across releases.

## Architecture

Everything lives in one publishable crate — the raw FFI layer and the safe
wrapper on top of it:

```
llamacpp-rs/
├── vendor/llama.cpp/          git submodule: upstream llama.cpp source (C/C++)
├── ffi/                       hand-written glue headers/sources bridging
│                               llama.cpp's C++ API to a flat C surface for bindgen
├── build.rs                    builds llama.cpp via cmake and runs bindgen over its
│                               headers, generating raw bindings into OUT_DIR
└── src/
    ├── llama_cpp_sys.rs        private module that includes the generated raw
    │                           `extern "C"` bindings (crate::llama_cpp_sys::*)
    └── ...                     model loading, context & KV-cache management,
                                 batching, sampling, grammars, chat templates,
                                 and multimodal (mtmd) support — the safe API
```

- `src/llama_cpp_sys.rs` is the FFI layer. It doesn't contain any llama.cpp
  source itself — instead the crate points at the upstream
  `ggml-org/llama.cpp` repo as a git submodule under `vendor/` (pinned to a
  fixed commit), compiles it with `cmake` in `build.rs`, and generates raw
  Rust bindings for its C API with `bindgen`. The hand-written glue that
  bridges llama.cpp's C++-only APIs (grammar, speculative decoding, mtmd) to
  a flat C surface lives under `ffi/`. Everything here is `unsafe` and maps
  almost 1:1 onto the C headers; it's a private module, so none of it is
  reachable from outside the crate.
- The rest of `src/` wraps those raw bindings in safe, ergonomic Rust types —
  handling lifetimes, ownership, and error handling so consumers never touch
  raw pointers. This is the public API users depend on.

This used to be split across two crates (`llama-cpp-sys` + `llamacpp-rs`,
the usual `-sys`/safe-wrapper pattern), but crates.io requires every
dependency of a published crate to itself already be published there, and
`llama-cpp-sys`'s vendored `llama.cpp` submodule made an independent release
more overhead than it was worth for a project that already doesn't follow
semver upstream. Merging into one crate keeps the module boundary (the FFI
layer stays `unsafe` and private to `llama_cpp_sys`) while leaving exactly
one crate to version and publish.

## Try it

llama.cpp is vendored as a submodule, so make sure it's checked out first
(see "Hacking" below).

Run the simple example (add `--features cuda` if you have a CUDA GPU):

```sh
cargo run --release -p simple -- --prompt "The way to kill a linux process is" hf-model TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
```

Run the multimodal example against an image (see `examples/mtmd`'s README
for the Windows/PowerShell form):

```sh
cargo run --release --example mtmd -- --model <model.gguf> --mmproj <mmproj.gguf> --image <photo.png> --prompt "Describe this image."
```

## Usage

Add the crate:

```sh
cargo add llamacpp-rs
```

Minimal end-to-end example — load a model, tokenize a prompt, and
greedily decode a response (see `examples/usage.rs` for the full runnable
version):

```rust
use llamacpp_rs::context::params::LlamaContextParams;
use llamacpp_rs::llama_backend::LlamaBackend;
use llamacpp_rs::llama_batch::LlamaBatch;
use llamacpp_rs::model::params::LlamaModelParams;
use llamacpp_rs::model::{AddBos, LlamaModel};
use llamacpp_rs::sampling::LlamaSampler;

let backend = LlamaBackend::init()?;
let model = LlamaModel::load_from_file(&backend, "model.gguf", &LlamaModelParams::default())?;
let mut ctx = model.new_context(&backend, LlamaContextParams::default())?;

let tokens = model.str_to_token("Hello! How are you?", AddBos::Always)?;
let mut batch = LlamaBatch::new(512, 1);
let last = tokens.len() as i32 - 1;
for (i, token) in (0_i32..).zip(tokens) {
    batch.add(token, i, &[0], i == last)?;
}
ctx.decode(&mut batch)?;

let mut sampler = LlamaSampler::greedy();
let token = sampler.sample(&ctx, batch.n_tokens() - 1);
if token != model.token_eos() {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    println!("{}", model.token_to_piece(token, &mut decoder, true, None)?);
}
```

From there:

- `run <cmd>` under `examples/simple`, `examples/embeddings`, `examples/mtmd`,
  and `examples/reranker` are full CLIs covering text generation, embeddings,
  multimodal (vision) input, and reranking respectively.
- Enable `--features sampler` for the wider sampler set (top-k, top-p, DRY,
  penalties, etc.) and `--features cuda`/`metal`/`vulkan`/`rocm`/`opencl` for
  GPU backends — see [Features](#features) below.
- Chat templates (`LlamaModel::apply_chat_template`), JSON-schema-constrained
  grammars (`json_schema_to_grammar`), and session/state save-load are all
  exposed as safe wrapper methods; see the crate's rustdoc
  (`cargo doc --open -p llamacpp-rs`) for the full API surface.

## Hacking

If you've already cloned without `--recursive`, pull in the vendored
llama.cpp submodule with:

```sh
git submodule update --init --recursive
```

## Testing

The full test suite covers unit tests, doctests, and integration tests that
exercise model loading, tokenization, batching, decoding, sampling,
KV-cache manipulation, and session/state save-load against a real (tiny)
GGUF model downloaded from Hugging Face on first run.

Run it directly (requires the llama.cpp submodule checked out and network
access on first run to fetch the tiny test model). This is the same command
CI runs as a merge gate on every push and pull request to `main` (see below):

```sh
./run-tests.sh              # fmt check + clippy + full test suite
./run-tests.sh --tests-only # skip fmt/clippy, just run the tests
```

On Windows, `run-tests.sh` is a bash script and won't run directly in
`powershell.exe` — invoke it through Git Bash instead. If plain `bash`
resolves to `C:\Windows\System32\bash.exe` (the WSL launcher) instead of
Git Bash, call Git Bash's `bash.exe` by its full path:

```powershell
& "C:\Program Files\Git\bin\bash.exe" run-tests.sh              # fmt check + clippy + full test suite
& "C:\Program Files\Git\bin\bash.exe" run-tests.sh --tests-only # skip fmt/clippy, just run the tests
```

### Coverage

What the suite in `llamacpp-rs/tests/` and inline unit tests (`#[cfg(test)]`)
actually exercises, and what's still untested. Update this list alongside
any change that adds or removes test coverage.

| Area | Covered? | Where / notes |
| --- | --- | --- |
| Model loading — valid model + metadata | ✅ | `model_loading.rs` |
| Model loading — error cases (missing file, junk bytes, renamed non-GGUF file, truncated header, empty/invalid/NUL-containing paths) | ✅ | `model_loading.rs` |
| Context params conversions and builder methods | ✅ | unit tests in `context/params.rs`, `context/params/get_set.rs` |
| Model params conversions and builder methods, incl. KV overrides | ✅ | unit tests in `model/params.rs`, `model/params/kv_overrides.rs` |
| Token encode/decode, token data, token data array, logit bias, token type | ✅ | unit tests in `token.rs` and submodules |
| GGUF metadata reading | ✅ | unit tests in `gguf/mod.rs` |
| Backend init/query | ✅ | unit tests in `llama_backend.rs` |
| Logging hooks | ✅ | unit tests in `log.rs` |
| Timing helpers | ✅ | unit tests in `timing.rs` |
| `json_schema_to_grammar` conversion | ✅ | unit test in `lib.rs` |
| Batch construction, capacity limits, decoding | ✅ | `batch_kv_session.rs` |
| KV-cache manipulation: seq copy/remove/shift/keep/clear | ✅ | `batch_kv_session.rs` |
| Session/state save-load round trip, incl. per-sequence state | ✅ | `batch_kv_session.rs` |
| Text generation / inference, incl. exact greedy output values | ✅ | `inference.rs` |
| Sampling: greedy determinism, logit bias, penalties, DRY sampler, sampler reset | ✅ | `sampling.rs` |
| Grammar-constrained sampling built without the `common` feature, incl. lazy patterns, invalid grammar, grammar-root checks | ✅ | `grammar_without_common.rs` |
| Chat template rendering, incl. `add_generation_prompt` and missing template-metadata error handling | ✅ | `chat_template.rs` |
| Embeddings: nonzero pooled output, determinism, distinctness across prompts | ✅ | `embeddings.rs` |
| mtmd (multimodal) pipeline end-to-end against a real vision model: init, bitmap decoding, tokenize with image chunks, eval, error cases for corrupt/truncated/missing image or mmproj input | ✅ | `mtmd_pipeline.rs` |
| Speculative decoding | ❌ | `src/speculative.rs` |
| `llguidance`-backed grammar sampling | ❌ | `src/llguidance_sampler.rs` |
| LoRA adapter init/set/remove | ❌ | `LlamaLoraAdapter*` in `context.rs`, `model.rs`, `lib.rs` |
| Encoder/decoder (`encode`) path for encoder-decoder models | ❌ | `EncodeError` in `lib.rs` |
| GPU backends (CUDA, Metal, Vulkan, ROCm, OpenCL) | ❌ | suite is CPU-only |
| Multi-sequence / multi-slot batching under concurrent decode | ❌ | — |

## Features

See `llamacpp-rs/Cargo.toml` for the full feature list (CUDA, Metal, Vulkan,
ROCm, OpenCL, static/dynamic linking options, etc.). `mtmd` (multimodal) and
`common` (JSON-schema-to-grammar helpers) are on by default.

## CI/CD

`.github/workflows/llama-cpp-rs-check.yml` builds and tests on
Linux/macOS/Windows/arm64 on every push and pull request to `main`, and
gates merges on the standalone test suite described above (`run-tests.sh`).

`.github/workflows/publish.yml` publishes `llamacpp-rs` to crates.io on
demand (`workflow_dispatch`) or when a `v*` tag is pushed, using a
`CARGO_REGISTRY_TOKEN` repo secret.

### Pre-release verification (2026-08-30)

Last full check before a crates.io release, on `feat/improve-test-coverage`,
after merging `llama-cpp-sys` into `llamacpp-rs` as a single publishable
crate:

| Check | Result |
| --- | --- |
| `cargo build --workspace` | ✅ success (only pedantic/deprecation warnings, no errors — same 18 warnings as before the merge) |
| `cargo test --workspace` | ✅ 90 lib/doc tests + all unit/integration tests passed, 0 failed (same counts as before the merge) |
| `cargo clippy --workspace --all-targets` | ✅ no errors, only pedantic-level style warnings |
| `cargo package -p llamacpp-rs` | ✅ packages cleanly (25.9MiB / 3.9MiB compressed) — no external path dependency, nothing blocking a publish |
| `cargo publish -p llamacpp-rs --dry-run` | ✅ builds and verifies from the packaged tarball successfully |

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
