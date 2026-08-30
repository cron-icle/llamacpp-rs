# Embeddings

Translation of llama.cpp's `embedding.cpp`: tokenizes one or more prompts
(one per line) and prints the pooled embedding vector for each.

## Usage

The prompt is a positional argument and must come *before* the model
subcommand (the subcommand consumes everything after it).

Using a local model file:

```sh
cargo run --release -p embeddings -- "Hello my name is" local ./bge-small-en-v1.5.Q4_K_M.gguf
```

Or download from Hugging Face automatically (cached after first run):

```sh
cargo run --release -p embeddings -- "Hello my name is" hf-model BAAI/bge-small-en-v1.5 BAAI-bge-small-v1.5.Q4_K_M.gguf
```

Pass a multi-line prompt to get one embedding per line, and `-n` to
normalize the output vectors:

**Linux/macOS:**

```sh
cargo run --release -p embeddings -- -n "$(printf 'first prompt\nsecond prompt')" local ./bge-small-en-v1.5.Q4_K_M.gguf
```

**Windows (PowerShell):**

```powershell
cargo run --release -p embeddings -- -n "first prompt`nsecond prompt" local ./bge-small-en-v1.5.Q4_K_M.gguf
```

Expected output: per-token tokenization is echoed to stderr, followed by
the embedding vector(s) and timing/throughput stats, e.g.:

```
n_ctx = 2048, n_ctx_train = 512

Prompt 0
...token dump...

Embeddings 0: [0.0123, -0.0456, ...]

Created embeddings for 6 tokens in 0.05 s, speed 120.00 t/s
```

### CLI arguments

- `local <path>` / `hf-model <repo> <file>` — model source (required, subcommand)
- `prompt` — prompt text, one embedding per line (default: `"Hello my name is"`)
- `-n` — normalize the produced embeddings
- `--disable-gpu` — run on CPU even when a GPU feature is enabled (requires `cuda`/`vulkan` feature)
