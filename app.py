import streamlit as st
from openai import OpenAI
import pandas as pd

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

# ---------- 📥 Load CSV Data ----------
@st.cache_data
def load_course_data():
    df = pd.read_csv('syracuse_ischool_courses.csv')  # Ensure this CSV is in your repo!
    return df

courses_df = load_course_data()

# ---------- 🔍 Search Function ----------
def search_courses(query):
    results = courses_df[courses_df['Course Name'].str.contains(query, case=False, na=False) |
                         courses_df['Description'].str.contains(query, case=False, na=False)]
    return results

# ---------- 💬 Chat Input ----------
user_input = st.chat_input("Ask me a course-related question...")

if user_input:
    # Step 1: Search CSV first
    matched_courses = search_courses(user_input)
    if not matched_courses.empty:
        st.write("📚 **Courses Found:**")
        for idx, row in matched_courses.iterrows():
            st.write(f"**{row['Course Name']}**: {row['Description']}")
    
    # Step 2: Fallback to OpenAI if no match OR general questions
    if client:
        st.session_state['history'].append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['history']
        )
        
        reply = response.choices[0].message.content.strip()
        
        st.session_state['history'].append({"role": "assistant", "content": reply})
        st.write(f"**Assistant**: {reply}")

# ---------- 📜 Display Chat History ----------
for msg in st.session_state['history']:
    role = msg['role'].capitalize()
    content = msg['content']
    st.write(f"**{role}**: {content}")
