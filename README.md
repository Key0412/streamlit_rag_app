# Simple RAG app

We'll create a simple RAG app that loads a PDF, chunks it, embeds it, stores in Faiss, and uses Ollama for generation. The UI will be via Streamlit.

This is a simple Retrieval-Augmented Generation (RAG) application using Langchain Classic running on Llama3.2.
Upload a PDF document and ask questions about its content.  

For a step by step on the rag application setup, see the notebook `./step_by_step_rag.ipynb`.

Cheatsheets and commands:

* https://docs.ollama.com/linux
* https://docs.ollama.com/cli
* Print GPU details: `nvidia-smi`
* Library of models: https://ollama.com/library?sort=popular

# Prepare Environment

## Python Packages

Packages:
- ollama: Provides tools for running and interacting with large language models locally, enabling model inference without relying on external APIs.
    - Ollama also needs the Ollama server to be running in the background for it to work properly. See the installation instructions in the `install_ollama.sh` file for details on how to set up the Ollama server.
- streamlit: Used for building interactive web applications, particularly for data science and machine learning projects.
- faiss: A library for efficient similarity search and clustering of dense vectors, commonly used in machine learning and information retrieval.
- langchain: A framework for integrating and managing language models, facilitating orchestration of workflows and interfacing with APIs such as OpenAI.
- langchain-community: LangChain Community contains third-party integrations that implement the base interfaces defined in LangChain Core, making them ready-to-use in any LangChain application.
- pypdf: A library for reading, manipulating, and extracting text from PDF files in Python.
- sentence-transformers: Provides tools for generating dense vector representations (embeddings) of sentences and texts, useful for semantic search, clustering, and similarity tasks.

Note:
- ollama focuses on local model inference and management.
- langchain is designed for workflow orchestration and integration with various language model APIs, including OpenAI.

## Installation

Setup python environment.
```
python3 -m venv <your_env_name>
source <your_env_name>/bin/activate
pip install -r requirements.txt
```

Install ollama cli:
```
./install_ollama.sh
```

Start Ollama in a shell:
```
ollama serve
```

Pull llama3.2 in another shell:
```
ollama pull llama3.2
```

# Run the app

`streamlit run rag_app.py`







