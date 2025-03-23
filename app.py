import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------- 🛠 Streamlit Page Config ----------
st.set_page_config(page_title="iSchool LLM Chatbot")
st.markdown("<h1 style='text-align: center; color: orange;'>Syracuse iSchool LLM Chatbot</h1>", unsafe_allow_html=True)

# ---------- 📥 Load Course Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv('syracuse_ischool_courses.csv')  # CSV should have Course Code, Course Name, Description
    return df

df = load_data()

# ---------- 🔥 Load Sentence-BERT Model ----------
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller + faster model, better for Streamlit Cloud
    return model

model = load_model()

# ---------- 🔥 Compute Embeddings ----------
@st.cache_data
def compute_embeddings(descriptions):
    return model.encode(descriptions, convert_to_numpy=True)

embeddings = compute_embeddings(df['Description'].tolist())

# ---------- 🔥 Build FAISS Index ----------
@st.cache_resource
def build_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

index = build_index(embeddings)

# ---------- 🔍 Search Function ----------
def search(query, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return indices[0]

# ---------- 💬 Chat ----------
user_input = st.chat_input("Ask about iSchool courses:")

if 'history' not in st.session_state:
    st.session_state['history'] = []

if user_input:
    st.session_state['history'].append({"role": "user", "content": user_input})
    indices = search(user_input)
    
    st.write("📚 **Courses Found:**")
    for idx in indices:
        course = df.iloc[idx]
        st.write(f"**{course['Course Code']} - {course['Course Name']}**: {course['Description']}")
    
    # Append assistant response (for history)
    st.session_state['history'].append({"role": "assistant", "content": f"Displayed top {len(indices)} course results."})

# ---------- 📜 Display Chat History ----------
if st.session_state['history']:
    st.write("---")
    st.write("### 📝 Chat History:")
    for msg in st.session_state['history']:
        role = msg['role'].capitalize()
        content = msg['content']
        st.write(f"**{role}**: {content}")
