import streamlit as st
import os
import tempfile
import pandas as pd
import json
import time
import plotly.express as px
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.vectorstores import FAISS
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader,
    JSONLoader, UnstructuredMarkdownLoader, UnstructuredPowerPointLoader,
    UnstructuredExcelLoader, UnstructuredHTMLLoader, UnstructuredEPubLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# 🔐 Gemini API Key
api_key = st.secrets["GEMINI_API_KEY"]

# --- Page Config and Custom CSS ---
st.set_page_config(page_title="DocGenius: Advanced Document Chat", page_icon="🤖", layout="wide")

# Custom CSS for a more professional look
st.markdown("""
    <style>
    .main { 
        background: linear-gradient(to right, #f8f9fa, #e9ecef); 
    }
    .stButton>button {
        background: linear-gradient(to right, #4CAF50, #2E8B57);
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .uploadedFile { 
        color: #4a4a4a; 
        font-weight: 600; 
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(255,255,255,0.7);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chat-bubble {
        padding: 12px 15px;
        margin: 10px 0;
        border-radius: 18px;
        position: relative;
        max-width: 80%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .chat-user {
        background: linear-gradient(135deg, #6A11CB 0%, #2575FC 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    .chat-bot {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-right: auto;
        border-bottom-left-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    div[data-testid="stSidebarNav"] li div a {
        margin-left: 1rem;
        padding: 1rem;
        width: 300px;
        border-radius: 0.5rem;
    }
    div[data-testid="stSidebarNav"] li div::focus-visible {
        background-color: rgba(151, 166, 195, 0.15);
    }
    div[data-testid="stMarkdownContainer"] h3 {
        padding-bottom: 1rem;
        border-bottom: 1px solid #e6e6e6;
    }
    .info-box {
        background-color: rgba(66, 135, 245, 0.1);
        border-left: 5px solid #4287f5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .success-box {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .metrics-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session States ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []
if "doc_summaries" not in st.session_state:
    st.session_state.doc_summaries = {}
if "doc_topics" not in st.session_state:
    st.session_state.doc_topics = {}
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0
if "tokens_used" not in st.session_state:
    st.session_state.tokens_used = 0
if "document_stats" not in st.session_state:
    st.session_state.document_stats = {
        "total_docs": 0,
        "total_chunks": 0,
        "total_tokens": 0,
        "doc_types": {}
    }
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---

def get_file_icon(file_type):
    icons = {
        "pdf": "📄",
        "txt": "📝", 
        "docx": "📃",
        "csv": "📊",
        "json": "🔄",
        "md": "📑",
        "pptx": "📙",
        "xlsx": "📈",
        "html": "🌐",
    }
    return icons.get(file_type, "📁")

def process_file(file):
    """Process a single file and return documents."""
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
            return None, f"Unsupported file type: {suffix}"
        
        documents = loader.load()
        
        # Update document stats
        if suffix not in st.session_state.document_stats["doc_types"]:
            st.session_state.document_stats["doc_types"][suffix] = 0
        st.session_state.document_stats["doc_types"][suffix] += 1
        st.session_state.document_stats["total_docs"] += 1
        
        return documents, None
    except Exception as e:
        return None, f"Error loading {file.name}: {str(e)}"
    finally:
        # Clean up the temp file
        os.unlink(tmp_path)

def split_documents(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    st.session_state.document_stats["total_chunks"] += len(chunks)
    # Estimate tokens (rough estimate)
    for chunk in chunks:
        st.session_state.document_stats["total_tokens"] += len(chunk.page_content.split()) * 1.3
    return chunks

def get_document_summary(file_name, docs):
    """Generate a summary for the document."""
    if not docs:
        return "No content available for summarization."
    
    # Combine content for summarization (limit size to avoid token issues)
    combined_text = " ".join([doc.page_content for doc in docs[:3]])
    if len(combined_text) > 5000:
        combined_text = combined_text[:5000] + "..."
    
    # Create a summary prompt
    summary_prompt = f"""
    Provide a brief summary (2-3 sentences) of the following document: {file_name}
    
    Content: {combined_text}
    
    Summary:
    """
    
    # Use the LLM to generate summary
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    summary = llm.invoke(summary_prompt)
    
    # Increment API call counter
    st.session_state.api_calls += 1
    
    return summary.content

def extract_document_topics(file_name, docs):
    """Extract key topics from the document."""
    if not docs:
        return []
    
    # Combine content for topic extraction (limit size to avoid token issues)
    combined_text = " ".join([doc.page_content for doc in docs[:3]])
    if len(combined_text) > 5000:
        combined_text = combined_text[:5000] + "..."
    
    # Create a topic extraction prompt
    topic_prompt = f"""
    Extract 3-5 key topics or themes from the following document: {file_name}
    
    Content: {combined_text}
    
    Return ONLY the list of topics, each as a single word or short phrase.
    """
    
    # Use the LLM to extract topics
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    topics_text = llm.invoke(topic_prompt)
    
    # Increment API call counter
    st.session_state.api_calls += 1
    
    # Parse the topics from the response
    topics = topics_text.content.split("\n")
    # Clean up the topics (remove numbering, bullet points)
    topics = [topic.strip().replace("- ", "").replace("* ", "") for topic in topics]
    topics = [topic for topic in topics if topic and len(topic) > 1]
    
    return topics[:5]  # Limit to 5 topics

def calculate_session_duration():
    """Calculate the session duration."""
    seconds = int(time.time() - st.session_state.start_time)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def create_document_visualizations():
    """Create visualizations based on document statistics."""
    # Document type distribution
    if st.session_state.document_stats["doc_types"]:
        doc_types = list(st.session_state.document_stats["doc_types"].keys())
        doc_counts = list(st.session_state.document_stats["doc_types"].values())
        
        # Create a pie chart using Plotly
        fig = px.pie(
            names=doc_types,
            values=doc_counts,
            title="Document Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(textinfo='percent+label', pull=[0.05] * len(doc_types))
        fig.update_layout(height=400)
        return fig
    return None

def generate_topics_wordcloud(topics_dict):
    """Generate a simple visualization of document topics."""
    if not topics_dict:
        return None
    
    # Flatten all topics
    all_topics = []
    for doc_topics in topics_dict.values():
        all_topics.extend(doc_topics)
    
    # Count topic frequency
    topic_freq = {}
    for topic in all_topics:
        if topic in topic_freq:
            topic_freq[topic] += 1
        else:
            topic_freq[topic] = 1
    
    # Generate simple bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    topics = list(topic_freq.keys())
    frequencies = list(topic_freq.values())
    
    # Sort by frequency
    sorted_indices = sorted(range(len(frequencies)), key=lambda i: frequencies[i], reverse=True)
    sorted_topics = [topics[i] for i in sorted_indices]
    sorted_frequencies = [frequencies[i] for i in sorted_indices]
    
    # Limit to top 10
    sorted_topics = sorted_topics[:10]
    sorted_frequencies = sorted_frequencies[:10]
    
    ax.barh(sorted_topics, sorted_frequencies, color='skyblue')
    ax.set_xlabel('Frequency')
    ax.set_title('Most Common Topics Across Documents')
    plt.tight_layout()
    
    # Convert to base64 for displaying
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return img_str

# --- Main App Logic ---
# App header with logo
col1, col2 = st.columns([1, 5])

with col1:
    st.image("https://img.icons8.com/llcolor/96/000000/brain--v2.png", width=80)
with col2:
    st.title("DocGenius: Advanced Document Chat")
    st.markdown("<p style='font-size: 1.2em;'>Intelligent conversation with your documents powered by Gemini + LangChain</p>", unsafe_allow_html=True)

# Create tabs for different app sections
tab1, tab2, tab3, tab4 = st.tabs(["📚 Document Upload", "💬 Chat Interface", "📊 Analytics", "⚙️ Settings"])

with tab1:
    st.header("Upload Your Documents")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "docx", "csv", "json", "md", "pptx", "xlsx", "html"],
            accept_multiple_files=True,
            key="file_uploader"
        )
    
    with col2:
        st.markdown("### Processing Options")
        chunk_size = st.slider("Chunk Size", min_value=500, max_value=3000, value=1500, step=100, 
                              help="Smaller chunks are better for specific questions, larger chunks preserve more context")
        
        overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=150, step=50,
                          help="Overlap between chunks helps maintain context across chunks")
    
    process_button = st.button("Process Documents", type="primary", use_container_width=True)
    
    if process_button and uploaded_files:
        with st.spinner("Processing documents... This may take a moment."):
            all_docs = []
            file_docs_map = {}  # Map file names to their documents
            
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                # Process each file
                docs, error = process_file(file)
                if error:
                    st.error(error)
                else:
                    # Store the documents for this file
                    file_docs_map[file.name] = docs
                    all_docs.extend(docs)
                    st.session_state.selected_files.append(file.name)
                    
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if all_docs:
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )
                chunks = text_splitter.split_documents(all_docs)
                
                # Create embeddings and vector store
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=api_key
                )
                
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.docs_processed = True
                
                # Generate summaries and extract topics for each file
                for file_name, docs in file_docs_map.items():
                    # Don't regenerate summaries for files we've already processed
                    if file_name not in st.session_state.doc_summaries:
                        st.session_state.doc_summaries[file_name] = get_document_summary(file_name, docs)
                        st.session_state.doc_topics[file_name] = extract_document_topics(file_name, docs)
                
                st.success(f"✅ Successfully processed {len(all_docs)} documents into {len(chunks)} chunks!")
                st.session_state.document_stats["total_chunks"] = len(chunks)
    
    # Display processed files
    if st.session_state.selected_files:
        st.markdown("### Processed Documents")
        
        for i, file_name in enumerate(st.session_state.selected_files):
            file_type = file_name.split(".")[-1].lower()
            icon = get_file_icon(file_type)
            
            with st.expander(f"{icon} {file_name}"):
                if file_name in st.session_state.doc_summaries:
                    st.markdown("#### Summary")
                    st.markdown(f"{st.session_state.doc_summaries[file_name]}")
                    
                    st.markdown("#### Key Topics")
                    if file_name in st.session_state.doc_topics:
                        topics = st.session_state.doc_topics[file_name]
                        topic_html = ""
                        for topic in topics:
                            topic_html += f'<span style="background-color: rgba(76, 175, 80, 0.2); padding: 5px 10px; margin: 0 5px 5px 0; border-radius: 15px; display: inline-block;">{topic}</span>'
                        st.markdown(f"{topic_html}", unsafe_allow_html=True)
                    else:
                        st.write("No topics extracted")
                        
                    if st.button(f"Remove {file_name}", key=f"remove_{i}"):
                        st.session_state.selected_files.remove(file_name)
                        if file_name in st.session_state.doc_summaries:
                            del st.session_state.doc_summaries[file_name]
                        if file_name in st.session_state.doc_topics:
                            del st.session_state.doc_topics[file_name]  
                        st.rerun()
                else:
                    st.info("Summary not available")

with tab2:
    if not st.session_state.docs_processed:
        st.info("Please upload and process documents in the 'Document Upload' tab first.")
    else:
        st.header("Chat with your documents")
        
        # Select chat model
        col1, col2 = st.columns([3, 1])
        with col2:
            model_option = st.selectbox(
                "Choose Gemini Model:",
                ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro","gemini-2.0-flash-lite","gemini-2.0-pro-exp-02-05",
"gemini-2.0-flash-thinking-exp-01-21","gemini-2.5-pro-exp-03-25","gemini-1.5-flash-8b"],
                help="Flash is faster, Pro is more capable for complex reasoning"
            )
        
        # Set up the chat container
        chat_container = st.container()
        
        with chat_container:
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            
            # Display chat messages
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"<div class='chat-bubble chat-user'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-bubble chat-bot'><strong>DocGenius:</strong> {message['content']}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Chat input
        st.markdown("### Ask a question about your documents")
        
        col1, col2 = st.columns([5, 1])
        with col1:
            user_question = st.text_input("", placeholder="Ask me anything about your documents...", key="user_question")
        with col2:
            clear_chat = st.button("🧹 Clear Chat")
            
        submit_question = st.button("Send", use_container_width=True, type="primary")
        
        if clear_chat:
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
            
        if submit_question and user_question:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Create the retriever
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            
            # Set up memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Add conversation history to memory
            for i in range(0, len(st.session_state.conversation_history), 2):
                if i+1 < len(st.session_state.conversation_history):
                    memory.chat_memory.add_user_message(st.session_state.conversation_history[i])
                    memory.chat_memory.add_ai_message(st.session_state.conversation_history[i+1])
            
            # Create the chain
            llm = ChatGoogleGenerativeAI(
                model=model_option,
                google_api_key=api_key,
                temperature=0.7,
                streaming=True
            )
            
            # Custom prompt
            prompt_template = ChatPromptTemplate.from_template("""
            Answer the following question based on the provided context. If the answer is not in the context, say "I don't have enough information to answer that" and suggest a related question that you might be able to answer.

            Context: {context}
            
            Chat History: {chat_history}
            
            Question: {question}
            
            Answer:
            """)
            
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_template}
            )
            
            # Process the user question with streaming response
            with st.spinner("Thinking..."):
                st_callback = StreamlitCallbackHandler(st.container())
                response = chain.invoke(
                    {"question": user_question},
                    callbacks=[st_callback]
                )
                
                # Add the answer to the chat
                answer = response["answer"]
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Add to conversation history
                st.session_state.conversation_history.append(user_question)
                st.session_state.conversation_history.append(answer)
                
                # Update stats
                st.session_state.api_calls += 1
                # Rough token estimation
                st.session_state.tokens_used += len(user_question.split()) + len(answer.split())
                
                # Rerun to refresh the UI
                st.rerun()

with tab3:
    st.header("Analytics Dashboard")
    
    if not st.session_state.docs_processed:
        st.info("Please upload and process documents to see analytics.")
    else:
        # Key metrics in a single row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
            st.metric("Documents Processed", st.session_state.document_stats["total_docs"])
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
            st.metric("Total Chunks", st.session_state.document_stats["total_chunks"])
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
            st.metric("API Calls", st.session_state.api_calls)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
            st.metric("Session Duration", calculate_session_duration())
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Document visualizations
        st.subheader("Document Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            doc_viz = create_document_visualizations()
            if doc_viz:
                st.plotly_chart(doc_viz, use_container_width=True)
            else:
                st.info("No document data available for visualization.")
                
        with col2:
            # Topic visualization
            if st.session_state.doc_topics:
                topics_img = generate_topics_wordcloud(st.session_state.doc_topics)
                if topics_img:
                    st.markdown("<h3>Common Topics</h3>", unsafe_allow_html=True)
                    st.image(f"data:image/png;base64,{topics_img}", use_container_width =True)
                else:
                    st.info("Not enough topic data for visualization.")
            else:
                st.info("No topic data available for visualization.")
        
        # Chat analysis
        st.subheader("Conversation Analysis")
        if st.session_state.conversation_history:
            # Count number of exchanges
            num_exchanges = len(st.session_state.conversation_history) // 2
            
            # Calculate average response length
            if num_exchanges > 0:
                response_lengths = [len(st.session_state.conversation_history[i].split()) 
                                   for i in range(1, len(st.session_state.conversation_history), 2)]
                avg_response_length = sum(response_lengths) / len(response_lengths)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                    st.metric("Total Exchanges", num_exchanges)
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                    st.metric("Avg Response Length (words)", int(avg_response_length))
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display a few recent conversations
                st.markdown("### Recent Conversations")
                for i in range(min(3, num_exchanges)):
                    idx = -2 * (i + 1)
                    q_idx = idx
                    a_idx = idx + 1
                    
                    if abs(q_idx) <= len(st.session_state.conversation_history) and abs(a_idx) <= len(st.session_state.conversation_history):
                        with st.expander(f"Conversation {num_exchanges - i}"):
                            st.markdown(f"**Question:** {st.session_state.conversation_history[q_idx]}")
                            st.markdown(f"**Answer:** {st.session_state.conversation_history[a_idx]}")
        else:
            st.info("No conversation data available for analysis.")

with tab4:
    st.header("Settings & Configuration")
    
    # Model settings
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        default_model = st.selectbox(
            "Default Chat Model",
            ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro","gemini-2.0-flash-lite","gemini-2.0-pro-exp-02-05",
"gemini-2.0-flash-thinking-exp-01-21","gemini-2.5-pro-exp-03-25","gemini-1.5-flash-8b"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Higher values make output more creative, lower values make it more deterministic"
        )
    
    with col2:
        chunk_count = st.slider(
            "Retrieved Chunks",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of document chunks to retrieve per query"
        )
        
        show_sources = st.checkbox(
            "Show Source Documents",
            value=True,
            help="Display the source documents used to generate responses"
        )
    
    # Advanced settings
    st.subheader("Advanced Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        enable_memory = st.checkbox(
            "Enable Long-term Memory",
            value=True,
            help="Allow the AI to remember previous interactions"
        )
        
        enable_streaming = st.checkbox(
            "Enable Response Streaming",
            value=True,
            help="Show responses as they are generated"
        )
    
    with col2:
        enable_analytics = st.checkbox(
            "Collect Usage Analytics",
            value=True,
            help="Collect anonymous usage data to improve the app"
        )
        
        debug_mode = st.checkbox(
            "Debug Mode",
            value=False,
            help="Show additional debug information"
        )
    
    # Export/Import options
    st.subheader("Data Management")
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "Export Chat History",
            data=json.dumps(st.session_state.conversation_history),
            file_name="docgenius_chat_history.json",
            mime="application/json",
        )
        
        st.download_button(
            "Export Document Summaries",
            data=json.dumps(st.session_state.doc_summaries),
            file_name="docgenius_summaries.json",
            mime="application/json",
        )
    
    with col2:
        uploaded_history = st.file_uploader(
            "Import Chat History",
            type=["json"],
            help="Upload a previously exported chat history"
        )
        
        if uploaded_history is not None:
            try:
                imported_history = json.loads(uploaded_history.read())
                if st.button("Load Imported History"):
                    st.session_state.conversation_history = imported_history
                    st.success("Chat history imported successfully!")
            except Exception as e:
                st.error(f"Error importing chat history: {e}")
    
    # Reset options
    st.subheader("Reset Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.success("Chat history cleared!")
    
    with col2:
        if st.button("Reset All Settings", type="secondary"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.success("All settings reset to default!")
            st.info("Please refresh the page to continue.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>DocGenius: Intelligent Document Chat | Powered by Gemini + LangChain</p>
    </div>
    """,
    unsafe_allow_html=True
)
