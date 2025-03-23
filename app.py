import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Streamlit config
st.set_page_config(page_title="iSchool LLM Chatbot")

st.markdown("<h1 style='text-align: center; color: orange;'>Syracuse iSchool LLM Chatbot</h1>", unsafe_allow_html=True)

# ---------- 📥 Load Data ----------
@st.cache_data
def load_data():
    return pd.read_csv('syracuse_ischool_courses.csv')

df = load_data()

# ---------- 🔥 Load Model ----------
@st.cache_resource
def load_model():
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    return model

model = load_model()

# ---------- 🔥 Compute Embeddings ----------
@st.cache_data
def compute_embeddings(descriptions):
    return model.encode(descriptions)

embeddings = compute_embeddings(df['Description'].tolist())

# ---------- 🔥 Build FAISS Index ----------
@st.cache_resource
def build_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

index = build_index(embeddings)

# ---------- 🔍 Search ----------
def search(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return indices[0]

# ---------- 💬 Chat ----------
user_input = st.chat_input("Ask about courses:")

if user_input:
    st.session_state.setdefault('history', []).append({"role": "user", "content": user_input})
    indices = search(user_input)
    
    st.write("📚 **Courses Found:**")
    for idx in indices:
        course = df.iloc[idx]
        st.write(f"**{course['Course Code']} - {course['Course Name']}**: {course['Description']}")
    
    for msg in st.session_state['history']:
        st.write(f"**{msg['role'].capitalize()}**: {msg['content']}")
