# filepath: /home/klismam/workspace/agent_study/rag_app.py
import streamlit as st
import os
import hashlib
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import tempfile

VECTOR_STORE_DIR = "faiss_indexes"


def get_store_path(content_hashes: list[str]) -> str:
    """Generate a unique FAISS store path based on the uploaded file *content* hashes."""
    key = "_".join(sorted(content_hashes))
    hashed = hashlib.md5(key.encode()).hexdigest()[:10]
    return os.path.join(VECTOR_STORE_DIR, hashed)


def handle_pdf_upload():
    """
    Handle (multiple) PDF file uploads via Streamlit UI.

    Returns:
        list of (original_name, temp_path, content_hash) triples, or empty list if none.
    """
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type="pdf", accept_multiple_files=True
    )

    results = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getbuffer()
            content_hash = hashlib.md5(file_bytes).hexdigest()[:12]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_bytes)
                results.append((uploaded_file.name, temp_file.name, content_hash))
    return results


@st.cache_resource
def load_documents(content_hashes_tuple, file_paths_tuple, chunk_size=500, chunk_overlap=50):
    """
    Load and split one or more PDF documents into chunks.

    Args:
        content_hashes_tuple (tuple): Content hashes of each PDF — used as the cache key.
        file_paths_tuple (tuple): Tuple of temp file paths to actually read from.
        chunk_size (int): Characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: LangChain Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_splits = []
    for temp_path in file_paths_tuple:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        splits = text_splitter.split_documents(documents)
        all_splits.extend(splits)
    return all_splits


@st.cache_resource
def create_vector_store(_documents, store_path, model_name="all-MiniLM-L6-v2"):
    """
    Create (or load from disk) a FAISS vector store for the given documents.

    Args:
        _documents (list): LangChain Document objects (prefixed with _ to skip cache hashing).
        store_path (str): Directory path for persisting this index (used as cache key).
        model_name (str): SentenceTransformer model name.

    Returns:
        FAISS: Vector store ready for similarity search.
    """
    model = SentenceTransformerEmbeddings(model_name=model_name)
    if os.path.exists(store_path):
        return FAISS.load_local(
            store_path, model, allow_dangerous_deserialization=True
        )
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vector_store = FAISS.from_documents(documents=_documents, embedding=model)
    vector_store.save_local(store_path)
    return vector_store


@st.cache_resource
def setup_rag_chain(_vector_store):
    """
    Set up a RAG chain that supports conversation history.

    Args:
        _vector_store (FAISS): Vector store with embedded documents.

    Returns:
        Chain: LangChain retrieval chain.
    """
    llm = Ollama(model="llama3.2")  # streaming is controlled via .stream() on the chain

    retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    # Prompt supports conversation history via chat_history placeholder
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the user's question based only on the "
            "context below. If the answer is not in the context, say so.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain


def build_chat_history() -> list:
    """Build a LangChain-compatible chat history from session state."""
    history = []
    for question, answer in st.session_state.get("interactions", []):
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
    return history


def stream_response(retrieval_chain, user_question: str) -> str:
    """
    Stream response tokens to the UI and return the full answer string.

    Args:
        retrieval_chain: The LangChain retrieval chain.
        user_question (str): The user's question.

    Returns:
        str: The complete answer text.
    """
    chat_history = build_chat_history()

    def token_generator():
        for chunk in retrieval_chain.stream(
            {"input": user_question, "chat_history": chat_history}
        ):
            if "answer" in chunk:
                yield chunk["answer"]

    answer = st.write_stream(token_generator())
    return answer


def main():
    """
    Main Streamlit application function.

    Orchestrates the RAG app: multi-PDF upload, vector store creation,
    conversational question answering with streaming, and history display.
    """
    st.title("RAG App with Langchain and Ollama — Llama3.2")
    st.write(
        "Upload one or more PDF documents and ask questions about their content. "
        "The assistant remembers your conversation history."
    )

    # ── Sidebar ─────────────────────────────────────────────────────────────
    st.sidebar.write("### Cache control")

    if st.sidebar.button("🔄 Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared!")

    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state["interactions"] = []
        st.rerun()

    # ── PDF Upload ───────────────────────────────────────────────────────────
    uploaded = handle_pdf_upload()  # list of (original_name, temp_path)

    if not uploaded:
        st.warning("Please upload at least one PDF file to proceed.")
        return

    file_names    = [name         for name, _, _hash in uploaded]
    temp_paths    = [temp         for _,    temp, _hash in uploaded]
    content_hashes = [_hash       for _,    _,    _hash in uploaded]

    st.sidebar.write("**Loaded PDFs:**")
    for name in file_names:
        st.sidebar.write(f"• {name}")

    # ── Index / Chain ────────────────────────────────────────────────────────
    store_path = get_store_path(content_hashes)          # keyed by content, not filename
    documents  = load_documents(
        tuple(content_hashes), tuple(temp_paths)         # cache keyed by content hash
    )
    vector_store = create_vector_store(documents, store_path)
    retrieval_chain = setup_rag_chain(vector_store)

    # ── Question Form ────────────────────────────────────────────────────────
    with st.form("question_form"):
        user_question = st.text_input("Ask a question about the document(s):")
        submitted = st.form_submit_button("Ask")

    if submitted and user_question:
        with st.spinner("Retrieving and generating answer..."):
            answer = stream_response(retrieval_chain, user_question)

        # Store after streaming so history is available for the next turn
        if "interactions" not in st.session_state:
            st.session_state["interactions"] = []
        st.session_state["interactions"].append((user_question, answer))

    # ── Conversation History ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Conversation History")

    interactions = st.session_state.get("interactions", [])
    if interactions:
        for i, (question, answer) in enumerate(reversed(interactions), 1):
            turn_number = len(interactions) - i + 1
            with st.container(border=True):
                st.write(f"**Q{turn_number}:** {question}")
                st.write(f"**A{turn_number}:** {answer}")
    else:
        st.info("No questions asked yet. Ask a question above to get started!")


if __name__ == "__main__":
    main()
