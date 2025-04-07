import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Page Configuration ---
st.set_page_config(page_title="💬 Persistent Chat with Memory", page_icon="💬", layout="wide")

st.markdown("""
    <style>
    .chat-bubble {
        padding: 12px;
        margin: 8px 0;
        border-radius: 10px;
    }
    .chat-user { background-color: #175673; color: white; }
    .chat-bot  { background-color: #3f0f4d; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("💬 Persistent Gemini Chat with Memory")
st.markdown("This Gemini remembers you're Nik during your chat session 🤖")

# --- API Key ---
api_key = st.secrets["GEMINI_API_KEY"]

# --- Gemini Model ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# --- Memory Setup ---
memory = ConversationBufferMemory(return_messages=True)

# --- Prompt Template with Name Instruction ---
template = """
You are Gemini, a helpful assistant. The user's name is Nik.
Always refer to them as Nik and never ask their name.

Chat history:
{history}
Nik: {input}
Gemini:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# --- Conversation Chain ---
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- UI Inputs ---
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("Ask something, Nik 👇", key="query_input")
with col2:
    clear = st.button("🧹 Clear Chat")

submit = st.button("Submit", use_container_width=True)

if submit and query:
    response = conversation.predict(input=query)
    st.session_state.chat_history.append((query, response))

if clear:
    st.session_state.chat_history = []
    memory.clear()

# --- Display Chat ---
for q, a in st.session_state.chat_history:
    st.markdown(f"<div class='chat-bubble chat-user'><strong>Nik:</strong> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble chat-bot'><strong>Gemini:</strong> {a}</div>", unsafe_allow_html=True)
