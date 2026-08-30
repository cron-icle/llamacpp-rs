# Simple

Translation of llama.cpp's `simple.cpp`: loads a model, tokenizes a prompt,
and greedily decodes a response token-by-token.

## Usage

Using a local model file:

```sh
cargo run --release -p simple -- --prompt "The way to kill a linux process is" local ./llama-2-7b.Q4_K_M.gguf
```

Or download from Hugging Face automatically (cached after first run):

```sh
cargo run --release -p simple -- --prompt "The way to kill a linux process is" hf-model TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
```

Expected output: the prompt tokens are echoed to stderr as they're
tokenized, then the generated continuation streams to stdout token-by-token
as it's decoded, followed by decode timing/throughput stats, e.g.:

```
n_len = 32, n_ctx = 2048, k_kv_req = 32

The way to kill a linux process is
 to send it a signal. The kill command sends a signal to a process...

decoded 24 tokens in 1.83 s, speed 13.11 t/s
```

### CLI arguments

- `local <path>` / `hf-model <repo> <file>` — model source (required, subcommand)
- `-p, --prompt <TEXT>` — prompt text (default: `"Hello my name is"`)
- `-f, --file <PATH>` — read the prompt from a file instead of `--prompt`
- `--n-len <N>` — total tokens (prompt + generated) to produce (default: `32`)
- `-c, --ctx-size <N>` — context window size (default: loaded from the model)
- `-s, --seed <N>` — RNG seed (default: `1234`)
- `-t, --threads <N>` / `--threads-batch <N>` — thread counts for generation / batch processing
- `-o KEY=VALUE` — override a model parameter (repeatable)
- `--main-gpu <N>` / `--devices <N,N,...>` — GPU selection (requires `cuda`/`vulkan` feature)
- `--disable-gpu` — run on CPU even when a GPU feature is enabled
- `--list-devices` — print available backend devices and exit
- `-v, --verbose` — enable llama.cpp logs
