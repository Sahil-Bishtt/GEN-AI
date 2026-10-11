# streamlit run main.py

import streamlit as st
import pypdf
from google import genai
from gtts import gTTS
import io
from pathlib import Path

# 1. Web Page Title & Config
st.set_page_config(
    page_title="CampusGPT | IEEE GenAI Build",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 CampusGPT")
st.caption("Your AI-powered campus information assistant")

# 2. Sidebar Inputs 
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter API Key", type="password")

    st.subheader("Available Documents")
    pdf_folder = Path(__file__).parent / "PDFs"
    pdf_files = sorted(pdf_folder.glob("*.pdf"))

    if pdf_files:
        for pdf in pdf_files:
            st.write(f"📄 {pdf.name}")
    else:
        st.warning("No PDFs found in the PDFs folder. Please create a 'PDFs' folder and add files.")

# Initialize Session States for memory and document storage
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# 3. Extract Text from PDF (Fixed loop and state assignment)
if pdf_files and not st.session_state.pdf_text:
    extracted_text = ""
    for pdf_path in pdf_files:
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    extracted_text += (
                        f"\n\n{text}\n"
                    )
        except Exception as e:
            st.sidebar.error(f"Error reading {pdf_path.name}: {e}")
    st.session_state.pdf_text = extracted_text

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
    
    elif not st.session_state.pdf_text.strip():
        warning_msg = "No readable PDF text found. Please check your PDFs folder."
        st.chat_message("assistant").warning(warning_msg)
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    
    else:
        with st.spinner("Analyzing your document..."):
            try:
                # Connect to the AI engine using the API key securely
                client = genai.Client(api_key=api_key)

                # Build conversation history string for multi-turn context
                history_str = ""
                for msg in st.session_state.messages[-5:-1]:
                    history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
                
                # Custom grounding prompt to ensure zero hallucination
                my_prompt = f"""
                You are CampusGPT, a friendly and reliable campus information assistant.

                You have access to the text extracted from multiple PDF documents.
                
                IMPORTANT RULES:
                
                1. GREETINGS AND CONVERSATION
                - Respond naturally to greetings, thanks, and farewells.
                - Examples: Hi, Hello, How are you, Thank you, Bye.
                - Do not require PDF evidence for ordinary social conversation.
                
                2. QUESTIONS ABOUT THE DOCUMENTS
                - Answer questions using the supplied document text.
                - Use information from ALL relevant documents, not just one.
                - Understand requests such as:

                "Summarize all PDFs"
                "Give me an overview of these documents"
                "What information do these files contain?"
                "Compare the course catalog and student handbook"
                - For a request to summarize all documents, provide a separate,
                concise summary of each document using its filename.
                - For a question about a specific topic, combine relevant
                information from different documents when appropriate.
                - You may summarize, paraphrase, explain, and organize facts
                without changing their meaning.
                - Do not invent facts or claim something is in a document
                unless supported by its text.
                
                3. WHEN INFORMATION IS MISSING
                - If the documents do not contain the answer, reply:
                "I'm sorry, but the information you're looking for is not available in the provided documents"
                - If part of the answer is available, provide that part and
                clearly explain what information is missing.
                - If documents contradict each other, explain the conflict.
                - Do not use outside knowledge to answer campus-specific
                factual questions.
                
                4. RESPONSE STYLE
                - Be friendly, clear, and concise.
                - Use headings and bullet points for summaries.
                - Mention the source filename when relevant.
                - For a summary of multiple PDFs, include the main purpose,
                important information, and key takeaways of each document.
                - Never say information is missing before checking the
                relevant supplied document text.
                
                CONVERSATION HISTORY:
                {history_str}

                DOCUMENT TEXT:
                --- START ---
                {st.session_state.pdf_text}
                --- END ---
                
                USER QUESTION:
                {user_query}
                
                Answer according to these rules.
                """
                
                # Send text over the network and get the reply (Using valid model ID)
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
