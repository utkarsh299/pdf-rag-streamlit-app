# --- Patches for SQLite on Streamlit Cloud ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ---------------------------------------------

import streamlit as st
import os
import shutil
import time
import tempfile

from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough

# --- App Configuration ---
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("📄 PDF RAG Assistant")
st.markdown("""
Welcome to the PDF RAG Assistant! Upload your PDFs, and I'll help you find answers within them.
This project is for demonstration purposes and uses free, open-source models.
""")

# --- Hugging Face Token and Model Selection ---
with st.sidebar:
    st.header("Configuration")
    # It's highly recommended to use st.secrets for the token in a real app
    # For this project, we'll use a text input for ease of use.
    hf_token = st.text_input("Enter your Hugging Face API Token:", type="password")
    
    st.markdown("### LLM Selection")
    llm_repo_id = st.selectbox(
        "Choose a Language Model:",
        ("google/gemma-2b-it", "mistralai/Mixtral-8x7B-Instruct-v0.1", "HuggingFaceH4/zephyr-7b-beta"),
        index=0
    )
    
    st.markdown("### Embedding Model")
    embedding_model_name = "BAAI/bge-small-en-v1.5"
    st.info(f"Using `{embedding_model_name}` for document embeddings.")

# --- File Uploader and Processing ---
uploaded_files = st.file_uploader(
    "Upload up to 10 PDF files", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("Please upload a maximum of 10 PDF files.")
        uploaded_files = None

# --- Main Logic: Processing and Chat ---
if uploaded_files and hf_token:
    
    # Use a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to the temporary directory
        for uploaded_file in uploaded_files:
            with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        # 1. Load and Chunk Documents
        with st.spinner("Loading and chunking documents..."):
            loader = PyPDFDirectoryLoader(temp_dir)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            st.success(f"Loaded and split {len(docs)} documents into {len(splits)} chunks.")

        # 2. Create Embeddings and Vector Store
        with st.spinner("Creating embeddings and vector store... This may take a moment."):
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            st.success("Vector store created successfully!")

        # 3. Initialize LLM and RAG Chain
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        llm = HuggingFaceHub(
            repo_id=llm_repo_id,
            model_kwargs={"temperature": 0.5, "max_new_tokens": 1024}
        )

        prompt_template = """
        Answer the question based only on the following context. 
        If you cannot find the answer in the context, say "I don't have enough information to answer that question."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        # 4. Chat Interface
        st.subheader("Ask a Question About Your PDFs")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("What is your question?"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_chain.invoke(query)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Clean up the vectorstore after use to save memory
        if st.button("Clear Chat and Database"):
            vectorstore.delete_collection()
            st.session_state.messages = []
            st.rerun()

else:
    st.info("Please upload PDF files and enter your Hugging Face token to begin.")
