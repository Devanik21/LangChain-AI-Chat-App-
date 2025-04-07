import streamlit as st
import google.generativeai as genai

# Streamlit App Title
st.set_page_config(page_title="Gemini Chatbot 💬✨", layout="wide")
st.title("💎 Gemini Chatbot")

# Sidebar - API Key Input
with st.sidebar:
    st.header("🔐 Gemini API Settings")
    api_key = st.text_input("Enter your Gemini API key:", type="password")
    if api_key:
        genai.configure(api_key=api_key)

        st.success("API key set successfully! 💖", icon="✅")
    else:
        st.warning("Please enter your Gemini API key to start chatting 🌸", icon="⚠️")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state and api_key:
    st.session_state.chat = genai.GenerativeModel("gemini-2.0-flash").start_chat(history=[])

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if api_key:
    user_prompt = st.chat_input("Type your message here 💬")
    if user_prompt:
        # Show user message
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Send to Gemini
        response = st.session_state.chat.send_message(user_prompt)
        reply = response.text

        # Show bot response
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
