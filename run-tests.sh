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

# Chat-templating (the entry point for chat completion) and embeddings are
# both user-facing features covered by their own end-to-end test files;
# `cargo test --features sampler` above already runs them as part of the
# workspace, called out explicitly here so their coverage is visible in CI
# output rather than buried in the aggregate run.
echo "==> cargo test -p llamacpp-rs --features sampler --test chat_template"
cargo test -p llamacpp-rs --features sampler --test chat_template

echo "==> cargo test -p llamacpp-rs --features sampler --test embeddings"
cargo test -p llamacpp-rs --features sampler --test embeddings

# The grammar-without-common tests are skipped unless LLAMA_TEST_VOCAB_GGUF
# points at a vocab GGUF file; wire up the one checked into the repo so this
# path is actually exercised by default.
echo "==> cargo test -p llamacpp-rs --no-default-features --features sampler,mtmd --test grammar_without_common"
LLAMA_TEST_VOCAB_GGUF="$(pwd)/llamacpp-rs/src/gguf/ggml-vocab-bert-bge.gguf" \
    cargo test -p llamacpp-rs --no-default-features --features sampler,mtmd --test grammar_without_common

# End-to-end mtmd (multimodal) pipeline tests against a real vision model.
# Downloads ~280MB (granite-docling-258M Q8_0 + mmproj) from Hugging Face on
# first run, cached by hf-hub afterward. Skip with SKIP_MTMD_PIPELINE=1 for
# fast local iteration or network-constrained environments.
if [[ "${SKIP_MTMD_PIPELINE:-}" != "1" ]]; then
    echo "==> cargo test -p llamacpp-rs --features sampler,mtmd --test mtmd_pipeline"
    cargo test -p llamacpp-rs --features sampler,mtmd --test mtmd_pipeline
else
    echo "==> skipping mtmd_pipeline (SKIP_MTMD_PIPELINE=1)"
fi

echo "==> all checks passed"
