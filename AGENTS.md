# AGENTS.md

Guidance for coding agents working in this repository.

## Project layout

- `llama-cpp-sys/` — low-level, unsafe FFI crate. Vendors upstream
  `ggml-org/llama.cpp` as a git submodule at `llama-cpp-sys/vendor/llama.cpp`,
  builds it via `cmake` in `build.rs`, and generates raw bindings with
  `bindgen`. Hand-written glue headers/sources that bridge llama.cpp's C++
  APIs to a flat C surface live under `llama-cpp-sys/ffi/`. Do not hand-edit
  generated bindings.
- `llama-cpp/` — safe, idiomatic Rust wrapper around the sys crate.
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
- Keep unsafe FFI concerns in `llama-cpp-rs-sys` and safe abstractions in
  `llama-cpp-rs` — don't leak raw pointers or unsafe APIs into the safe
  wrapper's public surface.
- Workspace lints (`missing_docs`, `missing_debug_implementations`,
  `clippy::pedantic`) are warnings — new public items should have docs.

## Verification

- `cargo build` / `cargo build --features cuda|metal|vulkan|rocm|opencl`
  as relevant to the change.
- `cargo test` for the `llama-cpp-rs` crate.
- `cargo run --release --bin simple -- ...` or the other examples for
  end-to-end checks against a real model when relevant.
