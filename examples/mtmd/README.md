# Rust mtmd-cli implementation

Partial port of the mtmd-cli.cpp example in the llama-cpp repository.

## Usage

### Command Line Interface

To run the mtmd example, you first need to download the model gguf file and the multimodal projection file, e.g. for Gemma3 you may use:

**Linux/macOS:**

```sh
wget https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf \
https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/mmproj-F16.gguf
```

**Windows (PowerShell):**

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf" -OutFile "gemma-3-4b-it-Q4_K_M.gguf"
Invoke-WebRequest -Uri "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/mmproj-F16.gguf" -OutFile "mmproj-F16.gguf"
```

To then run the example, provide an image file and run — a sample image
(`assets/turing.png`) is included so you can try it without sourcing your
own:

**Linux/macOS:**

```sh
cargo run --release --example mtmd -- \
  --model ./gemma-3-4b-it-Q4_K_M.gguf \
  --mmproj ./mmproj-F16.gguf \
  --image assets/turing.png \
  --prompt "What is in the picture?" \
  --no-gpu \
  --no-mmproj-offload \
  --marker "<start_of_image>"
```

**Windows (PowerShell):**

```powershell
cargo run --release --example mtmd -- `
  --model ./gemma-3-4b-it-Q4_K_M.gguf `
  --mmproj ./mmproj-F16.gguf `
  --image assets/turing.png `
  --prompt "What is in the picture?" `
  --no-gpu `
  --no-mmproj-offload `
  --marker "<start_of_image>"
```
