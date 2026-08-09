# llama-cpp-rs

A safe wrapper around [llama.cpp](https://github.com/ggml-org/llama.cpp) for Rust.

## Info

This crate wraps llama.cpp's C API closely, staying as up to date with
upstream llama.cpp as possible rather than building a heavier abstraction on
top of it.

## Dependencies

This uses `bindgen` (via `llama-cpp-rs-sys`) to build the bindings to
llama.cpp, so `clang` needs to be installed on your system. See
[bindgen's requirements](https://rust-lang.github.io/rust-bindgen/requirements.html)
for details.

## Disclaimer

This crate is *not safe* in the strict Rust sense — there are ways to misuse
the underlying llama.cpp API through it that can produce undefined behavior.
Do not use it for tasks where UB is unacceptable, and please open an issue
if you spot one.
