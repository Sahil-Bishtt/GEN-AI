import streamlit as st
import pypdf
from google import genai
from gtts import gTTS
import io

# 1. Web Page Title & Config
st.set_page_config(page_title="My First Year Chatbot", layout="centered")

st.title("My First Year Chatbot")

# 2. Sidebar Inputs 
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# Initialize Session States for memory and document storage
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# 3. Extract Text from PDF
if uploaded_file is not None and not st.session_state.pdf_text:
    try:
        reader = pypdf.PdfReader(uploaded_file)
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        st.session_state.pdf_text = extracted_text
        st.sidebar.success("PDF successfully processed and loaded!")
    except Exception as e:
        st.sidebar.error(f"Error reading PDF: {e}")

# Helper function to generate audio bytes for gTTS
def text_to_audio_bytes(text):
    tts = gTTS(text=text, lang='en')
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return audio_io

# 4. Display Past Chat History (Multi-turn support with Audio)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "audio_bytes" in message:
            st.audio(message["audio_bytes"], format="audio/mp3")

# 5. User Input & Processing
if user_query := st.chat_input("Ask a question"):
    
    # Display the user's question first
    st.chat_message("user").write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Check for empty variables
    if not api_key:
        error_msg = "Please enter your API Key in the Sidebar"
        st.chat_message("assistant").error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    elif not st.session_state.pdf_text:
        warning_msg = "Please upload an official knowledge-base PDF file in the sidebar first."
        st.chat_message("assistant").warning(warning_msg)
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    
    else:
        with st.spinner("AI is analyzing the document..."):
            try:
                # Connect to the AI engine using the API key securely
                client = genai.Client(api_key=api_key)

                # Build conversation history string for multi-turn context
                history_str = ""
                for msg in st.session_state.messages[-5:-1]:
                    history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
                
                # Custom grounding prompt to ensure zero hallucination
                my_prompt = (
                    "You are a strict data assistant. Answer the question using ONLY the provided text block below.\n"
                    "If the answer cannot be found in the text, reply exactly with: 'Information not found.'\n"
                    "Do not assume, extrapolate, or hallucinate any external facts.\n\n"
                    f"--- DATASET START ---\n{st.session_state.pdf_text}\n--- DATASET END ---\n\n"
                    f"Current Question: {user_query}"
                )
                
                # Send text over the network and get the reply
                response = client.models.generate_content(
                    model='gemini-3.5-flash-lite',
                    contents=my_prompt
                )
                
                # Output and store response with native Streamlit audio player
                answer = response.text
                audio_data = text_to_audio_bytes(answer)
                
                st.chat_message("assistant").write(answer)
                st.audio(audio_data, format="audio/mp3")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "audio_bytes": audio_data
                })
                            
            except Exception as e:
                error_msg = f"API Error: {e}"
                st.chat_message("assistant").error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
