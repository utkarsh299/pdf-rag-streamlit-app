# --- Patches for SQLite on Streamlit Cloud ---
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
from langchain_core.language_models.llms import LLM
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.pdf import PyMuPDFLoader


# --- Hugging Face Imports ---
from huggingface_hub import InferenceClient

# --- App Configuration ---
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("📄 PDF RAG Assistant")
st.markdown("Welcome! Upload your PDFs and ask questions about their content.")

# --- Session State Initialization ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = None

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input("Enter your Hugging Face API Token:", type="password")
    
    st.markdown("### LLM Selection")
    llm_repo_id = st.selectbox(
        "Choose a Language Model:",
        ("HuggingFaceH4/zephyr-7b-beta", "mistralai/Mixtral-8x7B-Instruct-v0.1", "google/gemma-2b-it"),
        index=0,
        help="Zephyr is recommended for speed and reliability."
    )
    
    st.markdown("### Embedding Model")
    embedding_model_name = "BAAI/bge-small-en-v1.5"
    st.info(f"Using `{embedding_model_name}` for embeddings.")

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Upload up to 10 PDF files", type="pdf", accept_multiple_files=True
)
if uploaded_files and len(uploaded_files) > 10:
    st.warning("Please upload a maximum of 10 PDF files.")
    uploaded_files = None

# --- PDF Processing Button ---
if uploaded_files:
    if st.button("Process PDFs"):
        # FIX #2: Clear old vectorstore from session state if it exists
        if st.session_state.vectorstore is not None:
            st.session_state.vectorstore.delete_collection()
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.last_retrieved_docs = None

        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            with st.spinner("Loading and chunking documents..."):
                loader = PyPDFDirectoryLoader(temp_dir)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)
            
            # with st.spinner("Loading and chunking documents..."):
            #     # --- START OF CHANGE ---
            #     docs = []
            #     for filename in os.listdir(temp_dir):
            #         if filename.endswith(".pdf"):
            #             loader = PyMuPDFLoader(os.path.join(temp_dir, filename))
            #             docs.extend(loader.load())
            #     # --- END OF CHANGE ---
                
            #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            #     splits = text_splitter.split_documents(docs)
            
            with st.spinner("Creating embeddings and vector store..."):
                embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
                # Save the new vectorstore in session state
                st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                st.success("PDFs processed and ready!")

# --- Main Chat Logic ---
if st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever()

    # --- Custom LLM Class (remains the same) ---
    class HuggingFaceChat(LLM):
        client: InferenceClient
        repo_id: str
        model_kwargs: dict

        def __init__(self, repo_id: str, token: str, model_kwargs: dict = None):
            super().__init__(
                client=InferenceClient(token=token),
                repo_id=repo_id,
                model_kwargs=model_kwargs or {}
            )

        @property
        def _llm_type(self) -> str: return "custom_huggingface_chat"
        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(
                messages=messages, model=self.repo_id, stream=False, **self.model_kwargs
            )
            return response.choices[0].message.content
        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            return {"repo_id": self.repo_id, "model_kwargs": self.model_kwargs}
    
    if hf_token:
        llm = HuggingFaceChat(
            repo_id=llm_repo_id, token=hf_token, model_kwargs={"max_tokens": 512}
        )
        
        # FIX #1: Improved, stricter prompt template
        prompt_template = """<|system|>
You are an expert assistant. Your task is to answer the user's question based only on the provided context.
- Be concise and answer directly.
- Do not add any introductory or concluding remarks.
- Do not explain your reasoning or mention the context in your answer.
- If the context does not contain the answer, state only: "The provided documents do not contain the answer to this question."</s>
<|user|>
Context:
{context}

Question: {question}</s>
<|assistant|>
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = {
            "context": retriever | format_docs, "question": RunnablePassthrough()
        } | prompt | llm

        st.subheader("Ask a Question About Your PDFs")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("What is your question?"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    retrieved_docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(query, k=5)
                    st.session_state.last_retrieved_docs = retrieved_docs_with_scores
                    
                    response = rag_chain.invoke(query)
                    st.markdown(response)
                    
            st.session_state.messages.append({"role": "assistant", "content": response})

        if st.session_state.last_retrieved_docs:
            if st.button("Inspect Last Retrieved Context"):
                st.subheader("Last Retrieved Context")
                for i, (doc, score) in enumerate(st.session_state.last_retrieved_docs):
                    st.markdown(f"**Chunk {i+1} (Score: {score:.4f})**")
                    st.info(f"Source: {doc.metadata.get('source', 'N/A')} | Page: {doc.metadata.get('page', 'N/A')}")
                    st.caption(doc.page_content)
    else:
        st.warning("Please enter your Hugging Face token in the sidebar to start chatting.")
