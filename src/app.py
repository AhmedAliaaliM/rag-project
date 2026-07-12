import streamlit as st
import httpx
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Research Assistant")
st.caption("Ask questions about your research papers — answers grounded in your documents.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieval candidates (top_k)", 5, 20, 10)
    rerank_top_n = st.slider("Re-rank keep (top_n)", 2, 6, 4)

    st.divider()
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Add a new PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Uploading..."):
            files = {"file": (uploaded_file.name,
                              uploaded_file.getvalue(),
                              "application/pdf")}
            try:
                r = httpx.post(f"{API_URL}/upload", files=files, timeout=30)
                if r.status_code == 200:
                    st.success(r.json()["message"])
                    st.info(r.json()["note"])
                else:
                    st.error(f"Upload failed: {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    # Health check
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            st.success("API connected ✅")
        else:
            st.error("API not responding")
    except:
        st.error("API offline — run `python src/api.py`")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Sources & Details"):
                st.write("**Sources:**", ", ".join(msg["sources"]))
                st.write("**Latency:**", f"{msg['latency']}s")

# Chat input
if question := st.chat_input("Ask a question about your research papers..."):
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating answer..."):
            try:
                response = httpx.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "top_k": top_k,
                        "rerank_top_n": rerank_top_n
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    latency = data["latency_seconds"]

                    st.markdown(answer)

                    with st.expander("📄 Sources & Details"):
                        st.write("**Sources:**", ", ".join(sources))
                        st.write("**Latency:**", f"{latency}s")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency": latency
                    })
                else:
                    st.error(f"API error: {response.text}")

            except httpx.TimeoutException:
                st.error("Request timed out — try a simpler question")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")