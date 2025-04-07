import streamlit as st
import os
import tempfile
import pandas as pd
import json

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader,
    JSONLoader, UnstructuredMarkdownLoader, UnstructuredPowerPointLoader,
    UnstructuredExcelLoader, UnstructuredHTMLLoader, UnstructuredEPubLoader
)

# 🔐 Gemini API Key
api_key = st.secrets["GEMINI_API_KEY"]

st.set_page_config(page_title="📄 Chat with Your Docs (LangChain + Gemini)", layout="wide")
st.title("📄 Chat with Your Docs using LangChain + Gemini")
st.markdown("Upload files (PDF, DOCX, TXT, CSV, JSON, MD, PPTX, XLSX, HTML) and ask anything about them!")


import pypandoc
pypandoc.download_pandoc()

# 🗂️ Upload section
uploaded_files = st.file_uploader(
    "Upload your documents", 
    type=["pdf", "txt", "docx", "csv", "json", "md", "pptx", "xlsx", "html"],
    accept_multiple_files=True
)

# 🌐 Process files
docs = []
if uploaded_files:
    for file in uploaded_files:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if suffix == "pdf":
                loader = PyPDFLoader(tmp_path)
            elif suffix == "txt":
                loader = TextLoader(tmp_path)
            elif suffix == "docx":
                loader = Docx2txtLoader(tmp_path)
            elif suffix == "csv":
                loader = CSVLoader(file_path=tmp_path)
            elif suffix == "json":
                loader = JSONLoader(file_path=tmp_path, jq_schema='.', text_content=False)
            elif suffix == "md":
                loader = UnstructuredMarkdownLoader(tmp_path)
            elif suffix == "pptx":
                loader = UnstructuredPowerPointLoader(tmp_path)
            elif suffix == "xlsx":
                loader = UnstructuredExcelLoader(tmp_path)
            elif suffix == "html":
                loader = UnstructuredHTMLLoader(tmp_path)
            else:
                continue

            docs.extend(loader.load())
        except Exception as e:
            st.warning(f"❌ Could not load {file.name}: {str(e)}")

    # 🔍 Embedding & Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 🤖 LLM & Memory
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # 💬 Chat Section
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about your documents:")
    if st.button("Submit") and query:
        result = chain({"question": query})
        answer = result["answer"]
        st.session_state.chat_history.append((query, answer))

    # 📝 Display conversation
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Gemini:** {a}")
else:
    st.info("📂 Upload some documents to get started.")
