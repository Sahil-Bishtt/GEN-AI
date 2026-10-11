import streamlit as st
import pypdf
from google import genai

# 1. Web Page Title
st.title("My first year Chatbot")

# 2. Sidebar Inputs 
with st.sidebar:
    api_key = st.text_input("Enter API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF")

# 3. Extract Text from PDF
pdf_text = ""
if uploaded_file is not None:
    try:
        reader = pypdf.PdfReader(uploaded_file)
        for page in reader.pages:
            # FIXED: Changed extra_text() to extract_text()
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"
    except Exception as e:
        st.sidebar.error(f"Error reading PDF: {e}")

# 4. User Input & Processing
if user_query := st.chat_input("Ask a question"):
    
    # Display the user's question first
    st.chat_message("user").write(user_query)
    
    # Check for empty variables
    if not api_key:
        st.chat_message("assistant").error("Please enter your API Key in the sidebar.")
    elif not pdf_text:
        st.chat_message("assistant").warning("Please upload a PDF file first.")
    else:
        with st.spinner("AI is reading the document..."):
            try:
                # Connect to the AI engine using the API key securely
                client = genai.Client(api_key=api_key)
                
                # Custom grounding prompt to ensure zero hallucination
                my_prompt = (
                    "You are a strict data assistant. Answer the question using ONLY the provided text block below.\n"
                    "If the answer cannot be found in the text, reply exactly with: 'Information not found.'\n\n"
                    f"--- DATASET START ---\n{pdf_text}\n--- DATASET END ---\n\n"
                    f"Question: {user_query}"
                )
                
                # Send text over the network and get the reply
                response = client.models.generate_content(
                    model='gemini-3.5-flash-lite',
                    contents=my_prompt
                )
                
                # Output the answer to the screen cleanly
                st.chat_message("assistant").write(response.text)
            except Exception as e:
                st.chat_message("assistant").error(f"API Error: {e}")
