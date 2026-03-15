#!/bin/bash
# Install Ollama on Linux AMD64
sudo apt-get install zstd
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst \
    | sudo tar x --zstd -C /usr
