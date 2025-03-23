import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Streamlit page configuration
st.set_page_config(page_title="iSchool LLM Chatbot")

st.markdown("<h1 style='text-align: center; color: orange;'>Syracuse iSchool LLM Chatbot</h1>", unsafe_allow_html=True)

# ---------- 📥 Load CSV Data ----------
@st.cache_data
def load_course_data():
    df = pd.read_csv('syracuse_ischool_courses.csv')  # Ensure this CSV is in your repo!
    return df

courses_df = load_course_data()

# ---------- BERT Model & FAISS Setup ----------

# Load pre-trained Sentence-BERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('bert-base-nli-mean-tokens')

model = load_model()

# Function to encode course descriptions into embeddings
@st.cache_data
def get_embeddings(texts):
    return model.encode(texts)

# Encode all course descriptions
course_embeddings = get_embeddings(courses_df['Description'].tolist())

# FAISS Indexing
@st.cache_resource
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity search
    index.add(np.array(embeddings))  # Add the embeddings to the index
    return index

faiss_index = build_faiss_index(course_embeddings)

# ---------- 🔍 Search Function using FAISS ----------
def search_courses(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    return indices[0], distances[0]

# ---------- 💬 Chat Input ----------
user_input = st.chat_input("Ask me a course-related question...")

if user_input:
    # Search for relevant courses
    indices, distances = search_courses(user_input)
    
    if len(indices) > 0:
        st.write("📚 **Courses Found:**")
        for idx in indices:
            course = courses_df.iloc[idx]
            st.write(f"**{course['Course Code']} - {course['Course Name']}**: {course['Description']}")
    else:
        st.write("😕 Sorry, I couldn't find any courses matching your query.")

# ---------- 📜 Display Chat History ----------
if 'history' not in st.session_state:
    st.session_state['history'] = []

st.session_state['history'].append({"role": "user", "content": user_input})

for msg in st.session_state['history']:
    role = msg['role'].capitalize()
    content = msg['content']
    st.write(f"**{role}**: {content}")
