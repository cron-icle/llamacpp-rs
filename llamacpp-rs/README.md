# llamacpp-rs

[![crates.io](https://img.shields.io/crates/v/llamacpp-rs.svg)](https://crates.io/crates/llamacpp-rs)
[![docs.rs](https://img.shields.io/docsrs/llamacpp-rs)](https://docs.rs/llamacpp-rs)
[![license](https://img.shields.io/crates/l/llamacpp-rs.svg)](#license)

Rust bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) — safe(ish) wrappers
around the C API, with multimodal (`mtmd`) support enabled by default.

This crate wraps llama.cpp's C API closely rather than building a heavier abstraction on
top of it. llama.cpp moves fast, so the goal here is to track upstream closely and stay
easy to keep in sync, rather than to offer a fully idiomatic, stable Rust API.

## Contents

- [Architecture](#architecture)
- [Install](#install)
- [Quick start](#quick-start)
- [Feature flags](#feature-flags)
- [Building from source](#building-from-source)
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)
- [License](#license)

## Architecture

llama.cpp is vendored as a submodule and compiled from source by `build.rs` (via `cmake`),
then bound into Rust with `bindgen`. A small C++ glue layer in `ffi/` fills gaps the raw
headers don't cover cleanly (e.g. batch helpers, mtmd plumbing), and the `src/` modules
wrap the generated FFI in safer, ergonomic types.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your application                         │
└───────────────────────────────┬───────────────────────────────┘
                                 │ safe Rust API
┌───────────────────────────────▼───────────────────────────────┐
│  llamacpp-rs (src/)                                              │
│  ┌───────────┐ ┌───────────┐ ┌────────────┐ ┌────────────────┐  │
│  │  model     │ │  context   │ │  sampling  │ │  token / gguf  │  │
│  │ (load,     │ │ (decode,   │ │ (samplers, │ │ (vocab, gguf   │  │
│  │  params)   │ │  kv-cache, │ │  grammars) │ │  metadata)     │  │
│  │            │ │  session)  │ │            │ │                │  │
│  └───────────┘ └───────────┘ └────────────┘ └────────────────┘  │
│  ┌────────────┐ ┌──────────────┐ ┌───────────────────────────┐  │
│  │ llama_batch │ │ mtmd (opt.)  │ │ speculative / timing / log │  │
│  └────────────┘ └──────────────┘ └───────────────────────────┘  │
└───────────────────────────────┬───────────────────────────────┘
                                 │ bindgen-generated FFI
┌───────────────────────────────▼───────────────────────────────┐
│  ffi/ (thin C/C++ glue: wrapper.h, wrapper_common.*,             │
│         wrapper_mtmd.h, wrapper_utils.h)                        │
└───────────────────────────────┬───────────────────────────────┘
                                 │
┌───────────────────────────────▼───────────────────────────────┐
│  vendor/llama.cpp (git submodule, built by build.rs via cmake)  │
│  llama-core, ggml (CPU/CUDA/Metal/Vulkan/ROCm/OpenCL backends),  │
│  common/ (grammar, sampling helpers), tools/mtmd                │
└───────────────────────────────────────────────────────────────┘
```

Key modules:

| Module | Purpose |
|---|---|
| `model` | Load a GGUF model (`LlamaModel::load_from_file`), tokenize/detokenize, inspect metadata |
| `context` | A model's inference context: decode/encode batches, KV cache, session save/restore |
| `llama_batch` | Build the token batches submitted to `decode`/`encode` |
| `sampling` | Samplers (greedy, top-k/top-p, mirostat, grammar-constrained, etc.) |
| `token` | Token data, token arrays, logit bias |
| `gguf` | Read GGUF file metadata directly |
| `mtmd` (feature `mtmd`) | Multimodal (image/audio) input pipeline |
| `speculative` (feature `common`) | Speculative decoding helpers |
| `grammar` | GBNF grammars for constrained generation |

## Install

```toml
[dependencies]
llamacpp-rs = "0.1"
```

You'll need `clang`/`cmake` available to build the vendored llama.cpp — see
[Building from source](#building-from-source).

## Quick start

```rust
use llamacpp_rs::llama_backend::LlamaBackend;
use llamacpp_rs::llama_batch::LlamaBatch;
use llamacpp_rs::model::params::LlamaModelParams;
use llamacpp_rs::model::{AddBos, LlamaModel};
use llamacpp_rs::context::params::LlamaContextParams;
use llamacpp_rs::sampling::LlamaSampler;

fn main() -> llamacpp_rs::Result<()> {
    let backend = LlamaBackend::init()?;
    let model = LlamaModel::load_from_file(&backend, "model.gguf", &LlamaModelParams::default())?;
    let mut ctx = model.new_context(&backend, LlamaContextParams::default())?;

    let tokens = model.str_to_token("Hello!", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1);
    let last = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens) {
        batch.add(token, i, &[0], i == last)?;
    }
    ctx.decode(&mut batch)?;

    let mut sampler = LlamaSampler::greedy();
    let token = sampler.sample(&ctx, batch.n_tokens() - 1);
    println!("{}", model.token_to_piece(token, &mut encoding_rs::UTF_8.new_decoder(), true, None)?);
    Ok(())
}
```

See the full runnable version in [`examples/usage.rs`](../examples/usage.rs):

```console
git clone --recursive https://github.com/cron-icle/llamacpp-rs
cd llamacpp-rs
wget https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_0.gguf
cargo run --example usage -- qwen2-1_5b-instruct-q4_0.gguf
```

There are also full CLI examples covering other workflows:

- [`examples/simple`](../examples/simple) — text generation
- [`examples/embeddings`](../examples/embeddings) — embeddings
- [`examples/mtmd`](../examples/mtmd) — multimodal (vision) input
- [`examples/reranker`](../examples/reranker) — reranking

## Feature flags

| Feature | Default | Description |
|---|:---:|---|
| `common` | ✅ | Links llama.cpp's `common/` static lib; enables JSON-schema→grammar and speculative decoding helpers |
| `mtmd` | ✅ | Multimodal (image/audio) input pipeline |
| `openmp` | ✅ | Enable OpenMP |
| `sampler` | | Rustier `sampler` struct in `context::sample::sampler` |
| `cuda` | | CUDA GPU backend |
| `cuda-no-vmm` | | CUDA without dynamically linking `libcuda` |
| `metal` | | No-op on non-Apple targets; Metal is always linked on Apple targets |
| `vulkan` | | Vulkan GPU backend |
| `rocm` | | ROCm GPU backend |
| `opencl` | | OpenCL backend (e.g. Adreno GPUs) |
| `mkl` | | Intel MKL BLAS backend (x86_64 only) |
| `dynamic-link` / `dynamic-backends` | | Dynamically link llama/ggml instead of statically |
| `system-ggml` / `system-ggml-static` | | Use a system-provided `ggml` instead of the vendored copy |
| `llguidance` | | Grammar-constrained sampling via [llguidance](https://github.com/guidance-ai/llguidance) |
| `static-stdcxx` / `shared-stdcxx` | | Control how the C++ stdlib is linked (mainly Android/Linux) |

## Building from source

This crate builds the vendored llama.cpp submodule from source using `cmake`, and uses
`bindgen` (in `build.rs`) to generate the raw FFI bindings against it and the glue in
`ffi/`. You'll need:

- `cmake`
- `clang` (see [bindgen's requirements](https://rust-lang.github.io/rust-bindgen/requirements.html))
- a C/C++ toolchain

Clone with submodules so the vendored llama.cpp source is present:

```console
git clone --recursive https://github.com/cron-icle/llamacpp-rs
```

## Disclaimer

This crate is *not safe* in the strict Rust sense — there are ways to misuse the
underlying llama.cpp API through it that can produce undefined behavior. Do not use it
for tasks where UB is unacceptable, and please open an issue if you spot one.

## Contributing

Issues and PRs are welcome. Since this crate tracks llama.cpp closely, changes that keep
the wrapper in sync with upstream API changes are especially useful.

## License

Licensed under either of [MIT](../LICENSE-MIT) or [Apache-2.0](../LICENSE-APACHE) at your option.
