import streamlit as st
from rag_engine import get_relevant_chunks
from llm_engine import ask_llm

st.title("🧠 Local RAG Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask something about the scans:", "")

if question:
    context = get_relevant_chunks(question)
    answer = ask_llm(question, context)
    st.session_state.history.append((question, answer))

for q, a in st.session_state.history[::-1]:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
