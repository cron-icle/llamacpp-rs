#!/usr/bin/env bash
# Standalone test-suite gate for llama-cpp-rs.
#
# Runs the same checks CI runs before a PR can merge: formatting, clippy, and
# the full test suite (unit + doc tests + integration tests: inference,
# batch/kv-cache/session, sampling, and grammar-without-common). Intended to
# be run either directly (with the toolchain and llama.cpp submodule already
# set up) or inside test.Dockerfile, which provides both.
#
# Usage:
#   ./run-tests.sh              # fmt check + clippy + full test suite
#   ./run-tests.sh --tests-only # skip fmt/clippy, just run tests
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${1:-}" != "--tests-only" ]]; then
    echo "==> cargo fmt --check"
    cargo fmt --check

    echo "==> cargo clippy --features sampler"
    cargo clippy --features sampler --all-targets
fi

echo "==> cargo test --features sampler (all workspace tests, incl. doctests)"
cargo test --features sampler

# The grammar-without-common tests are skipped unless LLAMA_TEST_VOCAB_GGUF
# points at a vocab GGUF file; wire up the one checked into the repo so this
# path is actually exercised by default.
echo "==> cargo test -p llamacpp-rs --no-default-features --features sampler,mtmd --test grammar_without_common"
LLAMA_TEST_VOCAB_GGUF="$(pwd)/llamacpp-rs/src/gguf/ggml-vocab-bert-bge.gguf" \
    cargo test -p llamacpp-rs --no-default-features --features sampler,mtmd --test grammar_without_common

echo "==> all checks passed"
