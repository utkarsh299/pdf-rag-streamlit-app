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
    # Use a session state variable to store the vectorstore
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if st.button("Process PDFs"):
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
                st.success("PDFs processed and ready!")

    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()

        # --- Custom LLM Class (Corrected) ---
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
                # This uses the official Zephyr chat template
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat_completion(
                    messages=messages,
                    model=self.repo_id,
                    stream=False,
                    **self.model_kwargs
                )
                return response.choices[0].message.content

            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                return {"repo_id": self.repo_id, "model_kwargs": self.model_kwargs}
        
        llm = HuggingFaceChat(
            repo_id=llm_repo_id,
            token=hf_token,
            model_kwargs={"max_tokens": 1024}
        )
        
        # --- NEW: RAG Chain that returns source documents ---
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # NEW: Zephyr's official prompt template
        prompt_template = """<|system|>
You are a helpful and concise assistant. Answer the user's question based only on the context provided. If the context does not contain the answer, say "I don't have enough information to answer that question."</s>
<|user|>
Context:
{context}

Question: {question}</s>
<|assistant|>
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        rag_chain = {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        } | prompt | llm

        # --- Chat Interface ---
        st.subheader("Ask a Question About Your PDFs")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "last_retrieved_docs" not in st.session_state:
            st.session_state.last_retrieved_docs = None

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("What is your question?"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve documents manually to get scores
                    retrieved_docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(query, k=5)
                    
                    # Store docs with scores in session state for the "Inspect" button
                    st.session_state.last_retrieved_docs = retrieved_docs_with_scores
                    
                    # Format the context for the LLM
                    context = format_docs([doc for doc, score in retrieved_docs_with_scores])
                    
                    # Invoke the chain with the manually retrieved context
                    response = rag_chain.invoke(query)
                    st.markdown(response)
                    
            st.session_state.messages.append({"role": "assistant", "content": response})

        # --- NEW: "Inspect Context" button ---
        if st.session_state.last_retrieved_docs:
            if st.button("Inspect Last Retrieved Context"):
                st.subheader("Last Retrieved Context")
                for i, (doc, score) in enumerate(st.session_state.last_retrieved_docs):
                    st.markdown(f"**Chunk {i+1} (Score: {score:.4f})**")
                    st.info(f"Source: {doc.metadata.get('source', 'N/A')} | Page: {doc.metadata.get('page', 'N/A')}")
                    st.caption(doc.page_content)

else:
    st.info("Please upload PDF files and enter your Hugging Face token to begin.")
