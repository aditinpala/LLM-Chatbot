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
def search_courses_faiss(query):
    query_embedding = get_bert_embeddings([query])
    _, indices = faiss_index.search(query_embedding, k=5)  # Search for top 5 most similar courses
    return courses_df.iloc[indices[0]]

# ---------- 💬 Chat Input ----------
user_input = st.chat_input("Ask me a course-related question...")

if user_input:
    # Step 1: Search Courses using FAISS
    matched_courses = search_courses_faiss(user_input)
    if not matched_courses.empty:
        st.write("📚 **Courses Found:**")
        for idx, row in matched_courses.iterrows():
            st.write(f"**{row['Course Name']}**: {row['Description']}")
    else:
        st.write("No matching courses found.")

# ---------- 📜 Display Chat History ----------
for msg in st.session_state.get('history', []):
    role = msg['role'].capitalize()
    content = msg['content']
    st.write(f"**{role}**: {content}")
