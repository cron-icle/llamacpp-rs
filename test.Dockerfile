# Self-contained, CPU-only environment for running llama-cpp-rs's test suite
# in isolation from the host machine. Requires the llama.cpp submodule to be
# checked out on the host before building the image (`git submodule update
# --init --recursive`), since Docker's build context can't fetch submodules
# itself.
#
# Build and run:
#   docker build -f test.Dockerfile -t llama-cpp-rs-test .
#   docker run --rm llama-cpp-rs-test
#
# Run a different command (e.g. just the new inference tests) against the
# same cached image:
#   docker run --rm llama-cpp-rs-test cargo test -p llama-cpp --features sampler --test inference
#
# The default command runs the same gate as CI: fmt check, clippy, and the
# full test suite (unit + doc + integration tests, including the tiny-model
# inference/batch/kv-cache/sampling/session tests, and the grammar tests
# built without the `common` feature).
ARG UBUNTU_VERSION=22.04
FROM ubuntu:${UBUNTU_VERSION}

# Requirements for rustup + bindgen: https://rust-lang.github.io/rust-bindgen/requirements.html
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && apt-get install -y \
    build-essential \
    curl \
    llvm-dev \
    libclang-dev \
    clang \
    pkg-config \
    libssl-dev \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y --profile minimal --component clippy,rustfmt
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /workspace
COPY . .

# Compile once at image build time so `docker run` starts from a warm
# incremental cache instead of rebuilding llama.cpp on every invocation.
RUN cargo build --features sampler

CMD ["./run-tests.sh"]
