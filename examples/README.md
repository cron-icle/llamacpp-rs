# Examples

Runnable CLIs demonstrating `llamacpp-rs`. Each is its own crate under
`examples/`; run any of them from the repo root with `cargo run --release
-p <crate> -- <args>` (or `--example <name>` for `mtmd`, see its README).

| Example | What it shows |
| --- | --- |
| [`simple`](simple) | Basic text generation: load a model, tokenize a prompt, greedily decode a response |
| [`embeddings`](embeddings) | Generate (and optionally normalize) pooled embeddings for one or more prompts |
| [`mtmd`](mtmd) | Multimodal (vision) input: describe an image with a vision-language model |
| [`reranker`](reranker) | Cross-encoder reranking of documents against a query |

All of them accept a model either as a local path (`local <path>`) or as a
Hugging Face repo/file pair that gets downloaded and cached (`hf-model
<repo> <file>`) — see each example's README for exact usage and sample
output.

Build with `--features cuda`/`vulkan`/`metal`/`rocm`/`opencl` (where the
example supports it) to offload to a GPU instead of running on CPU.
