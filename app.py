import streamlit as st
from openai import OpenAI

# Streamlit page config
st.set_page_config(page_title="iSchool LLM Chatbot")

st.markdown("<h1 style='text-align: center; color: orange;'>Syracuse iSchool LLM Chatbot</h1>", unsafe_allow_html=True)

# API Key input (secured)
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
client = None
if api_key:
    client = OpenAI(api_key=api_key)

# Tone Selection
tone = st.selectbox("Choose Bot Tone:", ["Formal Academic", "Friendly Casual"])

# Define System Prompt based on tone
if tone == "Formal Academic":
    system_prompt = "You are a formal academic advisor. Keep responses professional."
else:
    system_prompt = "You are a friendly study buddy. Keep responses casual."

# Initialize session state for memory
if 'history' not in st.session_state:
    st.session_state['history'] = [{"role": "system", "content": system_prompt}]

# Chat Input
user_input = st.chat_input("Ask me a course-related question...")

if user_input and client:
    # Append user message to history
    st.session_state['history'].append({"role": "user", "content": user_input})
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state['history']
    )
    
    # Get reply
    reply = response.choices[0].message.content.strip()
    
    # Append assistant reply to history
    st.session_state['history'].append({"role": "assistant", "content": reply})

# Display chat history nicely
for msg in st.session_state['history']:
    role = msg['role'].capitalize()
    content = msg['content']
    st.write(f"**{role}**: {content}")
