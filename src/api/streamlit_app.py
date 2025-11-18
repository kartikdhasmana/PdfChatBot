import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")
st.title("📄 PDF QA Bot with Memory")
st.markdown("Upload a PDF and ask questions based on the uploaded document. This bot remembers your **current session context.**")

# File uploader (commented out)
# uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
# if uploaded_file is not None:
#     with st.spinner("Uploading and processing PDF..."):
#         files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
#         response = requests.post(f"{API_URL}/upload", files=files)
#         if response.status_code == 200:
#             st.success(f"Uploaded and started processing: {uploaded_file.name}")
#         else:
#             st.error(f"Upload failed: {response.text}")

st.divider()

# Chat input
user_question = st.text_input("Ask a question:")
submit = st.button("Submit")

if submit and user_question.strip():
    with st.spinner("Thinking..."):
        try:
            payload = {"question": user_question, "top_k": 10}
            response = requests.post(f"{API_URL}/ask", json=payload)
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.success("✅ Answer:")
                st.markdown(answer)
            else:
                st.error(f"❌ Error: {response.text}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
