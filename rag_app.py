# filepath: /home/klismam/workspace/agent_study/rag_app.py
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic import hub
import tempfile

def handle_pdf_upload():
    """
    Handle PDF file upload via Streamlit UI.
    
    Displays a file uploader widget that accepts PDF files and saves them
    to a temporary location for processing.
    
    Returns:
        str or None: The file path to the uploaded PDF, or None if no file is uploaded.
    """
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        return temp_file_path
    return None

@st.cache_resource
def load_documents(file_path, chunk_size=500, chunk_overlap=50):
    """
    Load and split a PDF document into chunks.
    
    Loads a PDF file from the documents directory and splits it into smaller
    chunks for efficient processing by the embedding model.
    
    Args:
        file_path (str): The path to the PDF file to load.
        chunk_size (int, optional): The size of each text chunk in characters. Defaults to 500.
        chunk_overlap (int, optional): The overlap between consecutive chunks in characters. Defaults to 50.
    
    Returns:
        list: A list of LangChain Document objects representing the split text chunks.
    """
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(documents=documents)
    return splits

@st.cache_resource
def create_vector_store(documents, model_name="all-MiniLM-L6-v2"):
    """
    Create a FAISS vector store from document chunks.
    
    Generates embeddings for the provided documents using a SentenceTransformer model
    and stores them in a FAISS vector database for efficient similarity search.
    
    Args:
        documents (list): A list of LangChain Document objects to embed and store.
        model_name (str, optional): The name of the SentenceTransformer model to use. Defaults to "all-MiniLM-L6-v2".
    
    Returns:
        FAISS: A FAISS vector store containing the embedded documents.
    """
    # Using langchain's wrapper for sentence transformers and Storing the embeddings in a vector store
    model = SentenceTransformerEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(documents=documents, embedding=model)
    return vector_store

@st.cache_resource
def setup_rag_chain(_vector_store):
    """
    Set up a Retrieval-Augmented Generation (RAG) chain.
    
    Creates a complete RAG pipeline by combining a retriever from the vector store,
    an LLM (Ollama), and a prompt template to generate answers based on retrieved documents.
    Requires Ollama to be running locally with the llama3.2 model downloaded.
    
    Args:
        vector_store (FAISS): A FAISS vector store containing embedded documents.
    
    Returns:
        Chain: A LangChain retrieval chain that can process queries and generate answers.
    """
    # Use langchain's wrapper for Ollama
    # Dont forget to start your local Ollama server and have the model downloaded and running - see README.md for instructions
    llm = Ollama(model="llama3.2")  # Use your local Ollama model

    # Create a retriever with a score threshold
    retriever = _vector_store.as_retriever(
        kwargs={
            "search_type": "similarity",
            "search_kwargs": {"k": 10, "score_threshold": 0.5}
        }
    )

    # Fetch the prompt template for retrieval-based question answering
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Combine the llm model and the prompt template into a chain that can process retrieved documents and generate answers 
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    # Now, create the full retrieval chain by combining the retriever and the document processing chain
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

@st.cache_resource
def question_storage(user_question, response):
    """
    Store user questions in a cache resource.
    
    This function is designed to store the most recent user question in a cache resource,
    allowing it to be accessed across different runs of the Streamlit app. It can be used
    to maintain state or history of user interactions.
    
    Args:
        user_question (str): The user's question to be stored.
        response (str): The response to be stored.
    """
    if 'interactions' not in st.session_state:
        st.session_state['interactions'] = []
    st.session_state['interactions'].append((user_question, response))

def main():
    """
    Main Streamlit application function.
    
    Orchestrates the entire RAG application by initializing the UI, allowing users to
    select a PDF source (default or custom upload), loading and processing the document,
    and providing an interface for users to ask questions about the PDF content.
    """
    st.title("RAG App with Langchain and Ollama - Running on Llama3.2")
    st.write("""
             This is a simple Retrieval-Augmented Generation (RAG) application using Langchain Classic.  
             Upload a PDF document and ask questions about its content.
             """)
    
    # Option to upload a PDF or use the default file
    st.sidebar.write("### Cache control")
    
    # Clear cache button
    if st.sidebar.button("🔄 Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared!")
    
    # In the sidebar, after the "Clear Cache" button
    if st.sidebar.button("🗑️ Clear Conversation"):
        if 'interactions' in st.session_state:
            st.session_state['interactions'] = []
        st.rerun()
        
    pdf_path = handle_pdf_upload()
    if pdf_path is None:
        st.warning("Please upload a PDF file to proceed.")
        return
    
    # Load and process the document
    documents = load_documents(pdf_path)
    vector_store = create_vector_store(documents)
    retrieval_chain = setup_rag_chain(vector_store)

    # User input for questions
    with st.form("question_form"):
        user_question = st.text_input("Ask a question about the document:", )
        submitted = st.form_submit_button("Ask")
    
    if submitted and user_question:
        with st.spinner("Generating answer..."):
            response = retrieval_chain.invoke({"input": user_question})
            answer = response['answer']
            st.write(answer)
        question_storage(user_question, answer)
    
    # Display conversation history
    st.divider()
    st.subheader("Conversation History")
    
    if 'interactions' in st.session_state and st.session_state['interactions']:
        for i, (question, answer) in enumerate(st.session_state['interactions'][::-1], 1):
            if i==1:
                continue
            i = len(st.session_state['interactions']) - i + 1
            with st.container(border=True):
                st.write(f"**Q{i}:** {question}")
                st.write(f"**A{i}:** {answer}")
    else:
        st.info("No questions asked yet. Ask a question above to get started!")

if __name__ == "__main__":
    main()
