import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader, 
    CSVLoader, 
    JSONLoader, 
    UnstructuredMarkdownLoader, 
    UnstructuredPowerPointLoader, 
    UnstructuredExcelLoader, 
    UnstructuredHTMLLoader,
    UnstructuredEPubLoader
)

# Set page configuration
st.set_page_config(
    page_title="📄 Chat with Your Documents",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4285F4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5F6368;
        margin-bottom: 1.5rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message-user {
        background-color: #E8F0FE;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .chat-message-ai {
        background-color: #F1F3F4;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .stButton button {
        background-color: #4285F4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-header'>📄 Chat with Your Documents</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload files and ask questions about them using Gemini AI</p>", unsafe_allow_html=True)

# Sidebar for settings and file list
with st.sidebar:
    st.header("Settings")
    
    # API Key input
    api_key = st.text_input("Enter Gemini API Key:", type="password", 
                           value=st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else "")
    
    # Model selection
    model_option = st.selectbox(
        "Select Gemini Model:",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite", "gemini-2.0-pro-latest"]
    )
    
    # Retrieval settings
    k_docs = st.slider("Number of relevant chunks to retrieve:", min_value=1, max_value=10, value=3)
    
    # Clear buttons
    col1, col2 = st.columns(2)
    clear_chat = col1.button("Clear Chat")
    clear_files = col2.button("Clear Files")
    
    # Display uploaded files
    st.header("Uploaded Files")
    if "uploaded_file_names" in st.session_state:
        for file_name in st.session_state.uploaded_file_names:
            st.text(f"📄 {file_name}")
    else:
        st.info("No files uploaded yet")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# Clear chat if button pressed
if clear_chat:
    st.session_state.chat_history = []
    st.rerun()

# Clear files if button pressed
if clear_files:
    st.session_state.uploaded_file_names = []
    st.session_state.chain = None
    # Clean up temp files (would need to track them)
    st.rerun()

# File upload section
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload your documents", 
    type=["pdf", "txt", "docx", "csv", "json", "md", "pptx", "xlsx", "html", "epub"],
    accept_multiple_files=True
)
st.markdown("</div>", unsafe_allow_html=True)

# Process files and create chain
if uploaded_files and (st.session_state.chain is None or clear_files):
    with st.spinner("Processing documents... This might take a while for large files."):
        docs = []
        file_names = []
        
        for file in uploaded_files:
            file_names.append(file.name)
            suffix = file.name.split(".")[-1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            try:
                # Select the appropriate loader based on file type
                if suffix == "pdf":
                    loader = PyPDFLoader(tmp_path)
                elif suffix == "txt":
                    loader = TextLoader(tmp_path)
                elif suffix == "docx":
                    loader = Docx2txtLoader(tmp_path)
                elif suffix == "csv":
                    loader = CSVLoader(tmp_path)
                elif suffix == "json":
                    loader = JSONLoader(tmp_path, jq_schema='.', text_content=False)
                elif suffix == "md":
                    loader = UnstructuredMarkdownLoader(tmp_path)
                elif suffix == "pptx":
                    loader = UnstructuredPowerPointLoader(tmp_path)
                elif suffix == "xlsx":
                    loader = UnstructuredExcelLoader(tmp_path)
                elif suffix == "html":
                    loader = UnstructuredHTMLLoader(tmp_path)
                elif suffix == "epub":
                    loader = UnstructuredEPubLoader(tmp_path)
                else:
                    st.warning(f"Unsupported file type: {suffix}")
                    continue
                    
                docs.extend(loader.load())
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                continue
        
        # Store file names in session state
        st.session_state.uploaded_file_names = file_names
        
        if docs:
            try:
                # Create embeddings and vector store
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=api_key
                )
                vectorstore = FAISS.from_documents(docs, embeddings)
                
                # Set up LLM and chain
                llm = ChatGoogleGenerativeAI(
                    model=model_option, 
                    google_api_key=api_key,
                    temperature=0.2
                )
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                retriever = vectorstore.as_retriever(search_kwargs={"k": k_docs})
                
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm, 
                    retriever=retriever, 
                    memory=memory,
                    return_source_documents=True
                )
                
                st.session_state.chain = chain
                st.success(f"Successfully processed {len(docs)} document chunks from {len(file_names)} files!")
                
            except Exception as e:
                st.error(f"Error setting up the chat system: {str(e)}")
        else:
            st.warning("No documents were successfully processed.")

# Chat input area
st.subheader("💬 Ask about your documents")
query = st.text_input("Type your question here:", key="query_input")
send_button = st.button("Send")

# Process the query
if (send_button or query) and query.strip() and st.session_state.chain:
    with st.spinner("Thinking..."):
        try:
            result = st.session_state.chain({"question": query})
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Add to chat history
            st.session_state.chat_history.append((query, answer, source_docs))
            
            # Clear input
            st.session_state.query_input = ""
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            
elif (send_button or query) and not st.session_state.chain:
    st.warning("Please upload documents first before asking questions.")

# Display chat history
st.subheader("Conversation")
if not st.session_state.chat_history:
    st.info("Ask a question to start the conversation!")
else:
    chat_container = st.container()
    with chat_container:
        for i, (q, a, docs) in enumerate(st.session_state.chat_history):
            st.markdown(f"<div class='chat-message-user'><b>You:</b> {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-message-ai'><b>Gemini:</b> {a}</div>", unsafe_allow_html=True)
            
            # Optional: Show sources (can be toggled)
            with st.expander(f"View sources for answer #{i+1}"):
                for j, doc in enumerate(docs):
                    st.markdown(f"**Source {j+1}:**")
                    st.text(f"From: {getattr(doc.metadata, 'source', 'Unknown')}")
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.divider()

# Handle no API key
if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar to use this app.")

# Footer
st.markdown("---")
st.markdown("Powered by LangChain and Google Gemini API")
