# AGENTS.md

Guidance for coding agents working in this repository.

## Project layout

`llamacpp-rs/` is the single publishable crate — everything lives inside it:

- `llamacpp-rs/vendor/llama.cpp` — upstream `ggml-org/llama.cpp`, vendored
  as a git submodule.
- `llamacpp-rs/ffi/` — hand-written glue headers/sources that bridge
  llama.cpp's C++ APIs to a flat C surface for `bindgen`.
- `llamacpp-rs/build.rs` — builds llama.cpp via `cmake` and generates raw
  bindings with `bindgen` into `OUT_DIR`.
- `llamacpp-rs/src/llama_cpp_sys.rs` — private module that `include!`s the
  generated raw bindings (`crate::llama_cpp_sys::...`). Do not hand-edit
  generated bindings, and don't leak raw pointers or unsafe APIs from this
  module into the safe wrapper's public surface.
- `llamacpp-rs/src/*` (everything else) — the safe, idiomatic Rust wrapper.
  Most feature work (model loading, context/KV-cache, batching, sampling,
  grammars, chat templates, mtmd multimodal support) belongs here.
- `examples/` — runnable examples (`simple`, `embeddings`, `reranker`,
  `mtmd`) demonstrating the safe API.

See `README.md` for the full architecture overview.

## Setup

The llama.cpp submodule must be checked out before building:

```sh
git submodule update --init --recursive
```

## Conventions

- This project tracks llama.cpp closely and does not follow semver
  meaningfully.
- Workspace lints (`missing_docs`, `missing_debug_implementations`,
  `clippy::pedantic`) are warnings — new public items should have docs.
  The `llama_cpp_sys` module is exempted (`#![allow(...)]` at its top) since
  it's generated code.

## Verification

- `cargo build` / `cargo build --features cuda|metal|vulkan|rocm|opencl`
  as relevant to the change.
- `cargo test` for the `llamacpp-rs` crate.
- `cargo run --release --bin simple -- ...` or the other examples for
  end-to-end checks against a real model when relevant.
