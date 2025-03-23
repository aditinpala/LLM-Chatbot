import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import faiss
import numpy as np

# Streamlit page config
st.set_page_config(page_title="iSchool LLM Chatbot")

st.markdown("<h1 style='text-align: center; color: orange;'>Syracuse iSchool LLM Chatbot</h1>", unsafe_allow_html=True)

# ---------- 📥 Load CSV Data ----------
@st.cache_data
def load_course_data():
    df = pd.read_csv('syracuse_ischool_courses.csv')  # Ensure this CSV is in your repo!
    return df

courses_df = load_course_data()

# ---------- BERT Model & FAISS Setup ----------

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to encode course descriptions into embeddings
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use the embeddings of [CLS] token

# Encode all course descriptions
course_embeddings = get_bert_embeddings(courses_df['Description'].tolist())

# FAISS Indexing
faiss_index = faiss.IndexFlatL2(course_embeddings.shape[1])  # L2 distance for similarity search
faiss_index.add(np.array(course_embeddings))  # Add the embeddings to the index

# ---------- 🔍 Search Function using FAISS ----------
def search_courses(user_input):
    # Assuming you have a CSV of courses with columns 'Course Name' and 'Description'
    df = pd.read_csv("courses.csv")  # Make sure the path is correct
    matched_courses = df[df['Course Name'].str.contains(user_input, case=False, na=False)]
    return matched_courses


# ---------- 💬 Chat Input ----------
user_input = st.chat_input("Ask me a course-related question...")

if user_input:
    # Step 1: Search CSV first
    matched_courses = search_courses(user_input)
    if not matched_courses.empty:
        st.write("📚 **Courses Found:**")
        for idx, row in matched_courses.iterrows():
            # Display course code and name together
            st.write(f"**{row['Course Code']} - {row['Course Name']}**: {row['Description']}")
    
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
if 'history' in st.session_state:
    for msg in st.session_state['history']:
        role = msg['role'].capitalize()
        content = msg['content']
        st.write(f"**{role}**: {content}")

