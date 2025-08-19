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

# --- Hugging Face Imports ---
from huggingface_hub import InferenceClient, HfApi
from huggingface_hub.utils import HfHubHTTPError

# --- App Configuration ---
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("📄 PDF RAG Assistant")
st.markdown("Welcome! This app allows you to chat with your PDF documents using open-source models.")

# --- Session State Initialization ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = None
if "token_validated" not in st.session_state:
    st.session_state.token_validated = False
if "hf_token" not in st.session_state:
    st.session_state.hf_token = ""

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Step 1: Configuration")
    hf_token_input = st.text_input("Enter your Hugging Face API Token:", type="password", key="hf_token_input")
    
    if st.button("Validate Token"):
        if hf_token_input:
            try:
                # Use HfApi for validation, which has the whoami() method
                HfApi().whoami(token=hf_token_input)
                st.session_state.token_validated = True
                st.session_state.hf_token = hf_token_input
                st.success("Hugging Face token is valid!")
                st.rerun() # Immediately rerun to update the UI state
            except HfHubHTTPError:
                st.error("Invalid Hugging Face token. Please check and try again.")
                st.session_state.token_validated = False
        else:
            st.warning("Please enter a token before validating.")

    if st.session_state.token_validated:
        st.success("Token Validated!")
        st.markdown("---")
        st.markdown("### LLM Selection")
        llm_repo_id = st.selectbox(
            "Choose a Language Model:",
            ("HuggingFaceH4/zephyr-7b-beta", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
            index=0, help="Zephyr is recommended for speed and reliability."
        )
        embedding_model_name = "BAAI/bge-small-en-v1.5"
        st.info(f"Using `{embedding_model_name}` for embeddings.")
    else:
        st.warning("Please enter your token and click 'Validate Token' to proceed.")

# --- Main App UI ---

# Step 2: File Uploader
st.header("Step 2: Upload Your PDFs")
uploaded_files = st.file_uploader(
    "Upload up to 10 PDF files", 
    type="pdf", 
    accept_multiple_files=True,
    disabled=not st.session_state.token_validated
)
if uploaded_files and len(uploaded_files) > 10:
    st.warning("Please upload a maximum of 10 PDF files.")
    uploaded_files = None

# Step 3: Processing Button and Chat
st.header("Step 3: Process and Chat")
process_button_disabled = not (st.session_state.token_validated and uploaded_files)
if st.button("Process PDFs", disabled=process_button_disabled):
    if st.session_state.vectorstore is not None:
        st.session_state.vectorstore.delete_collection()
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
        with st.spinner("Creating embeddings and vector store..."):
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.success("PDFs processed and ready for chat!")

# Main Chat Logic (only runs if vectorstore is ready)
if st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever()

    class HuggingFaceChat(LLM):
        client: InferenceClient; repo_id: str; model_kwargs: dict
        def __init__(self, repo_id: str, token: str, model_kwargs: dict = None):
            super().__init__(client=InferenceClient(token=token), repo_id=repo_id, model_kwargs=model_kwargs or {})
        @property
        def _llm_type(self) -> str: return "custom_huggingface_chat"
        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(messages=messages, model=self.repo_id, stream=False, **self.model_kwargs)
            return response.choices[0].message.content
        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            return {"repo_id": self.repo_id, "model_kwargs": self.model_kwargs}
    
    llm = HuggingFaceChat(repo_id=llm_repo_id, token=st.session_state.hf_token, model_kwargs={"max_tokens": 512})
    
    prompt_template = """<|system|>
You are an expert assistant. Your task is to answer the user's question based only on the provided context.
- Be concise and answer directly.
- Do not add any introductory or concluding remarks.
- Do not explain your reasoning or mention the context in your answer.
- If the context does not contain the answer, state only: "The provided documents do not contain the answer to this question."</s>
<|user|>
Context: {context}
Question: {question}</s>
<|assistant|>
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm

    st.subheader("Ask a Question")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"): st.markdown(query)
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
