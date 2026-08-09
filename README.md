# llama-cpp-rs

Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp): a low-level
`llama-cpp-rs-sys` FFI crate and a safe `llama-cpp-rs` wrapper on top of it,
close to the underlying C API, with multimodal (`mtmd`) support enabled by
default so text, vision, and audio inputs all work out of the box.

This project tracks llama.cpp closely and does not follow semver
meaningfully — pin an exact version if you need stability across releases.

## Crates

- [`llama-cpp-rs`](llama-cpp-rs) — the safe wrapper. Model loading, context
  and KV-cache management, batching, sampling, grammars, chat templates, and
  multimodal (image/audio) input via `mtmd`.
- [`llama-cpp-rs-sys`](llama-cpp-rs-sys) — low-level bindgen-generated FFI
  bindings, vendoring and building llama.cpp itself.

## Try it

Clone the repo (with submodules — llama.cpp itself is vendored as one):

```bash
git clone --recursive <this-repo-url>
cd llama-cpp-rs
```

Run the simple example (add `--features cuda` if you have a CUDA GPU):

```bash
cargo run --release --bin simple -- --prompt "The way to kill a linux process is" hf-model TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
```

Run the multimodal example against an image:

```bash
cargo run --release -p mtmd -- --model <model.gguf> --mmproj <mmproj.gguf> --image <photo.png> --prompt "Describe this image."
```

## Hacking

If you've already cloned without `--recursive`, pull in the vendored
llama.cpp submodule with:

```sh
git submodule update --init --recursive
```

## Features

See each crate's `Cargo.toml` for the full feature list (CUDA, Metal,
Vulkan, ROCm, OpenCL, static/dynamic linking options, etc.). `mtmd`
(multimodal) and `common` (JSON-schema-to-grammar helpers) are on by
default.

## CI/CD

`.github/workflows/llama-cpp-rs-check.yml` builds and tests on
Linux/macOS/Windows/arm64 on every push and pull request to `main`.
`.github/workflows/publish-upon-release.yml` publishes both crates to
crates.io on a tagged GitHub release, gated on a `CARGO_REGISTRY_TOKEN`
repository secret.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
