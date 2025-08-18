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
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- Hugging Face Imports ---
from huggingface_hub import InferenceClient

# --- App Configuration ---
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("📄 PDF RAG Assistant")
st.markdown("""
Welcome! Upload your PDFs (up to 10) and ask questions. This app uses open-source models for processing.
""")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input("Enter your Hugging Face API Token:", type="password")
    
    st.markdown("### LLM Selection")
    llm_repo_id = st.selectbox(
        "Choose a Language Model:",
        ("HuggingFaceH4/zephyr-7b-beta", "mistralai/Mixtral-8x7B-Instruct-v0.1", "google/gemma-2b-it"),
        index=0,
        help="Zephyr is recommended as it's fast and reliable on the free tier."
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

# --- Main Logic: Processing and Chat ---
if uploaded_files and hf_token:
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        with st.spinner("Loading and chunking documents..."):
            loader = PyPDFDirectoryLoader(temp_dir)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            st.success(f"Loaded {len(docs)} documents, split into {len(splits)} chunks.")

        with st.spinner("Creating embeddings and vector store..."):
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            st.success("Vector store created!")

        # --- CORRECTED HuggingFaceChat Class ---
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
            def _llm_type(self) -> str:
                return "custom_huggingface_chat"

            def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat_completion(
                    messages=messages,
                    model=self.repo_id,
                    stream=False, # Use non-streaming response
                    **self.model_kwargs
                )
                # Extract the message content from the response object
                return response.choices[0].message.content

            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                return {"repo_id": self.repo_id, "model_kwargs": self.model_kwargs}
        # --- End of Correction ---

        llm = HuggingFaceChat(
            repo_id=llm_repo_id,
            token=hf_token,
            model_kwargs={"max_tokens": 1024}
        )

        prompt_template = """
        Answer the question based only on the following context:
        Context: {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

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
        
        if st.sidebar.button("Clear Chat and Start Over"):
            vectorstore.delete_collection()
            st.session_state.messages = []
            st.rerun()

else:
    st.info("Please upload PDF files and enter your Hugging Face token to begin.")
