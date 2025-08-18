# --- Patches for SQLite on Streamlit Cloud ---
# This is a common workaround for an issue with chromadb on Streamlit Community Cloud.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ---------------------------------------------

import streamlit as st
import os
import tempfile
from typing import Any, List, Mapping, Optional

# --- LangChain Imports ---
from langchain.prompts import ChatPromptTemplate
from langchain_core.llms import LLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough

# --- Hugging Face Imports ---
from huggingface_hub import InferenceClient

# --- App Configuration ---
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("📄 PDF RAG Assistant")
st.markdown("""
Welcome to the PDF RAG Assistant! Upload your PDFs, and I'll help you find answers within them.
This project is for demonstration purposes and uses free, open-source models deployed on Hugging Face.
""")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    # Using st.secrets is recommended for production, but text_input is fine for a demo project.
    hf_token = st.text_input("Enter your Hugging Face API Token:", type="password")
    
    st.markdown("### LLM Selection")
    llm_repo_id = st.selectbox(
        "Choose a Language Model:",
        ("google/gemma-2b-it", "mistralai/Mixtral-8x7B-Instruct-v0.1", "HuggingFaceH4/zephyr-7b-beta"),
        index=0,
        help="Select a model from the Hugging Face Hub to answer your questions."
    )
    
    st.markdown("### Embedding Model")
    embedding_model_name = "BAAI/bge-small-en-v1.5"
    st.info(f"Using `{embedding_model_name}` for document embeddings.")

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Upload up to 10 PDF files", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("Please upload a maximum of 10 PDF files.")
        uploaded_files = None

# --- Main Logic: Processing and Chat ---
if uploaded_files and hf_token:
    
    # Use a temporary directory for processing uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            # Write each uploaded file to the temporary directory
            with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        # 1. Load and Chunk Documents
        with st.spinner("Loading and chunking documents... This may take a moment."):
            loader = PyPDFDirectoryLoader(temp_dir)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            st.success(f"Loaded and split {len(docs)} documents into {len(splits)} chunks.")

        # 2. Create Embeddings and Vector Store
        with st.spinner("Creating embeddings and vector store... This is usually the longest step."):
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            st.success("Vector store created successfully!")

        # 3. Initialize LLM and RAG Chain (with custom LLM class)
        # Define a custom LLM wrapper for the modern Hugging Face Inference API
        class HuggingFaceInference(LLM):
            repo_id: str
            client: InferenceClient
            model_kwargs: dict

            def __init__(self, repo_id: str, token: str, model_kwargs: dict = None):
                super().__init__()
                self.repo_id = repo_id
                self.client = InferenceClient(model=repo_id, token=token)
                self.model_kwargs = model_kwargs or {}

            @property
            def _llm_type(self) -> str:
                return "custom_huggingface_inference"

            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                **kwargs: Any,
            ) -> str:
                # The modern client uses the text_generation method
                response = self.client.text_generation(prompt, **self.model_kwargs)
                return response

            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                """Get the identifying parameters."""
                return {"repo_id": self.repo_id, "model_kwargs": self.model_kwargs}

        # Initialize our custom LLM
        llm = HuggingFaceInference(
            repo_id=llm_repo_id,
            token=hf_token,
            model_kwargs={"temperature": 0.5, "max_new_tokens": 1024, "return_full_text": False}
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
        
        # Add a button to clear the chat and database
        if st.sidebar.button("Clear Chat and Start Over"):
            vectorstore.delete_collection()
            st.session_state.messages = []
            st.rerun()

else:
    st.info("Please upload PDF files and enter your Hugging Face token to begin.")
