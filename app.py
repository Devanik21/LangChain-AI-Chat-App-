import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Page Configuration ---
st.set_page_config(page_title="💬 Continuous Chat App with Memory", page_icon="💬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 8px;
        font-size: 16px;
    }
    .chat-bubble {
        padding: 12px;
        margin: 8px 0;
        border-radius: 10px;
    }
    .chat-user {
        background-color: #175673;
        color: white;
    }
    .chat-bot {
        background-color: #3f0f4d;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title & Description ---
st.title("💬 Continuous Chat App with Memory")
st.markdown("Chat with Gemini and retain conversation context within your session.")

# 🔐 API Key from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]

# ✅ Setup LLM and Memory (with correct key: 'history')
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=api_key)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory)

# 🔄 Chat session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 💬 Chat UI
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("Nik, ask your question:", key="query_input")
with col2:
    clear = st.button("🧹 Clear Chat")

submit = st.button("Submit", use_container_width=True)

if submit and query:
    response = conversation.predict(input=query)
    st.session_state.chat_history.append((query, response))

if clear:
    st.session_state.chat_history = []
    memory.clear()

# 📜 Display conversation
for q, a in st.session_state.chat_history:
    st.markdown(f"""
        <div class='chat-bubble chat-user'><strong>Nik:</strong> {q}</div>
        <div class='chat-bubble chat-bot'><strong>Gemini:</strong> {a}</div>
    """, unsafe_allow_html=True)
