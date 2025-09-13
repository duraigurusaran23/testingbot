import streamlit as st
from dotenv import load_dotenv
import asyncio
import nest_asyncio
import os
import time
import shutil
from pathlib import Path
from datetime import datetime
import pickle
import json

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Custom loaders
from chunk_loader import load_processed_chunks
from scraper import WebsiteScraper

# -------------------------------
# Setup
# -------------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

nest_asyncio.apply()
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

API_KEYS = [
    'AIzaSyDDEBszT3jzbD7X5DQaG0VyEgOuGINEU4M',
    'AIzaSyAD8hLxcdnrC-_lZr4eJPNU_VvGfrbFPBU',
    'AIzaSyBTjNLOgw-KWyYVto1qUnZkxxrUbjCJkMk',
    'AIzaSyAx6kiWvvKGrGaHAf1e1pCI8CKDgLvFSEU'
]

if "current_key_index" not in st.session_state:
    st.session_state.current_key_index = 0

GOOGLE_API_KEY = API_KEYS[st.session_state.current_key_index]

st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")

# -------------------------------
# Session State Initialization
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "bot", "text": "👋 Hi! Enter `new + url` to start."}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "current_url" not in st.session_state:
    st.session_state.current_url = None
if "scraping" not in st.session_state:
    st.session_state.scraping = False
if "scraping_url" not in st.session_state:
    st.session_state.scraping_url = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# -------------------------------
# Utility Functions
# -------------------------------
def add_message(role, text):
    st.session_state["messages"].append({"role": role, "text": text})

def delete_existing_data():
    pdfs_dir = Path("../pdfs")
    processed_dir = Path("../processed_data")
    if pdfs_dir.exists():
        shutil.rmtree(pdfs_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    pdfs_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)

def switch_key():
    st.session_state.current_key_index = (st.session_state.current_key_index + 1) % len(API_KEYS)
    global GOOGLE_API_KEY
    GOOGLE_API_KEY = API_KEYS[st.session_state.current_key_index]
    st.cache_resource.clear()
    docs = get_docs()
    if docs:
        st.session_state.qa_chain = get_chain("gemini-1.5-flash", docs)

def update_env_with_current_key():
    with open('.env', 'w') as f:
        f.write(f'GOOGLE_API_KEY={API_KEYS[st.session_state.current_key_index]}\n')
        for i, key in enumerate(API_KEYS):
            if i != st.session_state.current_key_index:
                f.write(f'# {key}\n')

# -------------------------------
# Chatbot Initialization
# -------------------------------
@st.cache_data(show_spinner=False)
def get_docs():
    """Load pre-processed chunks"""
    try:
        docs = load_processed_chunks()
        return docs if docs else []
    except Exception as e:
        return []

@st.cache_resource
def get_chain(selected_model, docs):
    if not docs:
        return None
    try:
        # Try local embeddings first
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            embedding_provider = "Local (Sentence Transformers)"
        except Exception as local_e:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            embedding_provider = "Google Gemini"

        vector_store = FAISS.from_texts(docs, embeddings)
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 50, "fetch_k": 100})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        custom_prompt_template = """
         # 🤖 Advanced Website Knowledge Assistant

         You are an intelligent AI assistant specialized in website content analysis and Q&A. Your primary function is to help users understand and extract information from scraped website data.

         ## 🎯 CORE PRINCIPLES

         ### 1. CONTEXT-ONLY ANSWERING
         - **MANDATORY**: Answer ONLY using the provided website context
         - **NEVER** use external knowledge, assumptions, or general information
         - **DEFAULT RESPONSE** for unavailable information: "I don't know. The information is not available in the provided website content."
         - **VERIFICATION**: Cross-reference information across multiple context chunks when possible

         ### 2. INTELLIGENT CONTENT RECOGNITION
         **Synonyms & Related Terms:**
         - **Leadership**: CEO, Founder, Owner, President, Director, Executive, Head, Boss, Manager, Chairperson
         - **Team/People**: Employees, Staff, Workers, Team members, Personnel, Associates, Colleagues, Workforce
         - **Company**: Organization, Firm, Business, Corporation, Enterprise, Startup, Agency, Consultancy
         - **Products/Services**: Solutions, Offerings, Features, Capabilities, Platforms, Tools, Systems
         - **Contact**: Reach out, Get in touch, Connect, Email, Phone, Message, Contact form
         - **Skills/Abilities**: Expertise, Proficiency, Knowledge, Experience, Qualifications, Competencies
         - **Certifications**: Certificates, Credentials, Qualifications, Awards, Achievements, Accreditations
         - **Education**: Degree, Course, Training, Learning, Academic background, Qualifications
         - **Experience**: Work history, Career, Professional background, Tenure, Employment

         ### 3. SPECIALIZED CONTENT HANDLING

         **Skills Analysis:**
         - List ALL technical skills mentioned (HTML, CSS, JavaScript, Python, AWS, React, Node.js, etc.)
         - Include proficiency levels when specified (Beginner, Intermediate, Advanced, Expert)
         - Group related skills (Frontend: HTML/CSS/JS, Backend: Python/Node.js, Cloud: AWS/GCP)
         - Mention tools, frameworks, and technologies

         **Certifications & Qualifications:**
         - List complete certification names with providers
         - Include dates, validity periods, and credential IDs when available
         - Group by category (Technical, Professional, Industry-specific)
         - Note any specializations or concentrations

         **Company/Service Information:**
         - Extract mission, vision, values, and company culture
         - Identify products, services, and target markets
         - Note unique selling propositions and competitive advantages
         - Include pricing, packages, or service tiers when mentioned

         ### 4. CONVERSATION INTELLIGENCE

         **Chat History Integration:**
         - Reference previous questions and answers for context
         - Understand follow-up questions and clarifications
         - Maintain conversation flow and topic continuity
         - Avoid repeating information already provided

         **Question Interpretation:**
         - Recognize implicit questions and requests
         - Handle multi-part questions systematically
         - Provide comprehensive answers for broad queries
         - Ask for clarification only when absolutely necessary

         ### 5. RESPONSE OPTIMIZATION

         **Content Structure:**
         - **Headers**: Use clear, descriptive headers for organized responses
         - **Lists**: Use bullet points or numbered lists for multiple items
         - **Tables**: Use markdown tables for comparisons or structured data
         - **Code**: Use code blocks for technical content, commands, or examples

         **Professional Communication:**
         - Use business-appropriate, professional language
         - Be concise yet comprehensive - no unnecessary verbosity
         - Maintain consistent tone throughout responses
         - Use active voice and clear sentence structure

         **Error Handling:**
         - Gracefully handle incomplete or unclear context
         - Provide partial information when available
         - Suggest related topics or alternative questions
         - Maintain helpful attitude even with limitations

         **Greeting Handling:**
         - For very short inputs that are purely greetings like "hi", "hello", "hey", "good morning", respond with a friendly receptionist-style greeting
         - Examples: "Hello! How can I help you today?" or "Hi there! What can I assist you with regarding the company?"
         - For questions or longer inputs, always answer based on context
         - Do not provide company information in greetings unless specifically asked
         - Keep responses warm and welcoming only for clear greetings

         ## 📋 RESPONSE GUIDELINES


         ### For Company Information:
         ## Company Overview
         [Company Name] is a [industry] company specializing in [services/products].

         ## Key Services
         - Service 1: [Description]
         - Service 2: [Description]

         ## Unique Value Proposition
         [What makes them stand out]
         ```

         ### For General Questions:
         - Provide direct, factual answers
         - Include relevant context and details
         - Reference specific sections or pages when possible
         - Maintain objectivity and accuracy

         ## 🔍 CONTEXT ANALYSIS FRAMEWORK

         When analyzing context:
         1. **Identify Key Sections**: Headers, navigation, main content areas
         2. **Extract Structured Data**: Lists, tables, specifications
         3. **Note Relationships**: How different pieces of information connect
         4. **Prioritize Relevance**: Focus on information most relevant to the question
         5. **Maintain Accuracy**: Only include information explicitly stated

         ## 🚀 ADVANCED FEATURES

         - **Multi-page Synthesis**: Combine information from different website sections
         - **Temporal Awareness**: Note dates, timelines, and chronological information
         - **Comparative Analysis**: Handle questions comparing different options or services
         - **Requirements Matching**: Help users find services/products that match their needs

         ---

         **Context**: {context}
         **Chat History**: {chat_history}
         **Question**: {question}

         **Answer**:
         """
        CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_prompt_template)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGoogleGenerativeAI(
                model=selected_model,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.0,
                max_tokens=300,
                max_retries=1
            ),
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT}
        )

        st.session_state.embedding_provider = embedding_provider
        return qa_chain
    except Exception as e:
        st.error(f"Failed to initialize the chatbot: {str(e)}")
        return None

# -------------------------------
# User Input Handling
# -------------------------------
def handle_user_query(user_input, retry_count=0):
    if not user_input.strip():
        return

    # Clear chat history if user inputs "clear"
    if user_input.strip().lower() == "clear":
        st.session_state.chat_history = []
        st.session_state["messages"] = [{"role": "bot", "text": "👋 Hi! Enter `new + url` to start."}]
        add_message("bot", "Chat history cleared.")
        return

    if retry_count == 0:  # Only add message on first attempt
        add_message("user", user_input)

    # Start new scraping if user inputs "new + url"
    if user_input.lower().startswith("new"):
        url_part = user_input[4:].strip().lstrip("+").strip()
        if url_part.startswith("http"):
            st.session_state.scraping = True
            st.session_state.scraping_url = url_part
        else:
            add_message("bot", "⚠️ Please provide a valid URL starting with http:// or https://")
        return

    # If data not processed yet
    if not st.session_state.processed:
        add_message("bot", "Please scrape a website first by typing 'new + URL'")
        return

    # Query the QA chain
    try:
        result = st.session_state.qa_chain.invoke({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })
        answer = result.get("answer", "I couldn't generate a response.")
        if retry_count == 0:
            add_message("bot", answer)
        else:
            add_message("bot", f"Switched to new API key. {answer}")
        st.session_state.chat_history.append((user_input, answer))
        update_env_with_current_key()  # Update .env with working key
    except Exception as e:
        error_str = str(e).lower()
        if ("rate limit" in error_str or "quota" in error_str or "resource exhausted" in error_str) and retry_count < len(API_KEYS) - 1:
            print("api key rate finished")
            switch_key()
            handle_user_query(user_input, retry_count + 1)
        else:
            add_message("bot", f"⚠️ Error generating response: {str(e)}")

# -------------------------------
# Scraping Logic
# -------------------------------
if st.session_state.scraping and st.session_state.scraping_url:
    url = st.session_state.scraping_url
    delete_existing_data()
    scraper = WebsiteScraper()
    chunks = scraper.scrape_to_chunks(url, lambda p,s: None, lambda: False)

    processed_dir = Path("../processed_data")
    processed_dir.mkdir(exist_ok=True)
    metadata = {
        "processing_date": datetime.now().isoformat(),
        "source_url": url,
        "total_chunks": len(chunks),
        "total_characters": sum(len(c) for c in chunks)
    }

    with open(processed_dir / "chunks.json", 'w', encoding='utf-8') as f:
        json.dump({"chunks": chunks, "metadata": metadata}, f, indent=2, ensure_ascii=False)
    with open(processed_dir / "chunks.pkl", 'wb') as f:
        pickle.dump({"chunks": chunks, "metadata": metadata}, f)

    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.processed = True
    st.session_state.current_url = url
    st.session_state.scraping = False
    # Clear previous company history when new URL is entered
    st.session_state.chat_history = []
    st.session_state["messages"] = []
    add_message("bot", f"✅ Successfully scraped and processed {url}. You can now ask questions!")
    st.rerun()

# -------------------------------
# Load docs and initialize QA chain
# -------------------------------
docs = get_docs()
if docs:
    st.session_state.qa_chain = get_chain("gemini-1.5-flash", docs)

# -------------------------------
# Professional Page UI
# -------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* Global Styles - 8-point grid system */
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f7fa;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #000000;
}

/* Main App Container */
.stApp {
    background: transparent !important;
    max-width: 100% !important;
}

/* Center everything perfectly */
.main .block-container {
    padding: 0 !important;
    max-width: none !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    min-height: 100vh !important;
}

/* Modern Chat Container */
.chat-container {
    width: 448px; /* 56 * 8px */
    height: 672px; /* 84 * 8px */
    background: #FFFFFF;
    border-radius: 24px; /* 3 * 8px */
    border: 1px solid #e1e5e9;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1), 0 8px 16px rgba(0, 0, 0, 0.06);
}

/* Header */
.chat-header {
    padding: 24px; /* 3 * 8px */
    background: #f8fafc;
    border-bottom: 1px solid #e1e5e9;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px; /* 1.5 * 8px */
}

.chat-title {
    font-size: 22px; /* Header title size */
    font-weight: 600; /* Semi-Bold */
    margin: 0;
    color: #1a202c;
}

.bot-icon {
    width: 32px; /* 4 * 8px */
    height: 32px; /* 4 * 8px */
    background: #007BFF;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    color: #ffffff;
    box-shadow: 0 2px 8px rgba(0, 123, 255, 0.2);
}

/* Scrollable Messages Area */
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px; /* 3 * 8px */
    scroll-behavior: smooth;
    display: flex;
    flex-direction: column;
    gap: 16px; /* 2 * 8px */
}

/* Message Bubbles */
.message {
    display: flex;
    align-items: flex-end;
    gap: 8px; /* 1 * 8px */
    animation: slideIn 0.3s ease-out;
    width: 100%;
}

.message.bot {
    justify-content: flex-start;
}

.message.user {
    justify-content: flex-end;
}

.message-avatar {
    width: 32px; /* 4 * 8px */
    height: 32px; /* 4 * 8px */
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}

.message.bot .message-avatar {
    background: #F0F2F5;
    color: #666666;
}

.message.user .message-avatar {
    background: #007BFF;
    color: #ffffff;
}

.message-bubble {
    max-width: 320px; /* 40 * 8px */
    padding: 12px 16px; /* 1.5 * 8px, 2 * 8px */
    border-radius: 16px; /* 2 * 8px */
    font-size: 16px; /* Chat message text size */
    line-height: 1.4;
    word-wrap: break-word;
    position: relative;
    color: #000000;
}

.message.bot .message-bubble {
    background: #F0F2F5;
    border-bottom-left-radius: 4px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.message.user .message-bubble {
    background: #D6EFFF;
    border-bottom-right-radius: 4px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

/* Input Footer */
.input-footer {
    padding: 24px; /* 3 * 8px */
    background: #f8fafc;
    border-top: 1px solid #e1e5e9;
}

.input-row {
    display: flex;
    gap: 16px; /* 2 * 8px */
    align-items: center;
}

.input-row .stTextInput > div > div > input {
    border-radius: 24px !important; /* 3 * 8px */
    border: 2px solid #d1d5db !important;
    padding: 12px 20px !important; /* 1.5 * 8px, 2.5 * 8px */
    font-size: 16px !important;
    background: #ffffff !important;
    color: #000000 !important;
    transition: all 0.2s ease !important;
    flex: 1 !important;
}

.input-row .stTextInput > div > div > input::placeholder {
    color: #9ca3af !important;
}

.input-row .stTextInput > div > div > input:focus {
    border-color: #007BFF !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1) !important;
}

.input-row .stButton > button {
    width: 48px !important; /* 6 * 8px */
    height: 48px !important; /* 6 * 8px */
    border-radius: 50% !important;
    background: #007BFF !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3) !important;
    font-size: 20px !important;
    line-height: 1 !important;
    padding: 0 !important;
}

.input-row .stButton > button:hover {
    background: #0056CC !important;
    transform: scale(1.05) !important;
    box-shadow: 0 6px 16px rgba(0, 123, 255, 0.4) !important;
}

.input-row .stButton > button:active {
    transform: scale(0.95) !important;
}

/* Scrollbar Styling */
.chat-messages::-webkit-scrollbar {
    width: 8px; /* 1 * 8px */
}

.chat-messages::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}

.chat-messages::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 4px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.5);
}

/* Animations */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Hide Streamlit elements */
.stTextInput, .stButton, .stForm {
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Responsive Design */
@media (max-width: 480px) {
    .chat-container {
        width: 95vw !important;
        height: 85vh !important;
        margin: 16px !important; /* 2 * 8px */
    }

    .chat-header {
        padding: 16px !important; /* 2 * 8px */
    }

    .chat-title {
        font-size: 18px !important;
    }

    .chat-messages {
        padding: 16px !important; /* 2 * 8px */
    }

    .input-footer {
        padding: 16px !important; /* 2 * 8px */
    }
}
</style>
""", unsafe_allow_html=True)

# Header - Always visible
st.markdown("""
    <div class="chat-header">
        <div class="bot-icon">🤖</div>
        <h1 class="chat-title">AI Chat Assistant</h1>
    </div>
""", unsafe_allow_html=True)

# Messages Area
st.markdown('<div class="chat-messages" id="chat-messages">', unsafe_allow_html=True)

for msg in st.session_state["messages"]:
    role_class = "bot" if msg["role"] == "bot" else "user"
    avatar_icon = "🤖" if msg["role"] == "bot" else "👤"
    st.markdown(f"""
        <div class="message {role_class}">
            <div class="message-avatar">{avatar_icon}</div>
            <div class="message-bubble">{msg['text']}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    st.markdown('<div class="input-row">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 0.1])
    with col1:
        user_input = st.text_input(
            "Message",
            key="user_input_box",
            placeholder="Ask me anything...",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("✈️")
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted and user_input:
        handle_user_query(user_input)
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Auto-scroll JavaScript
st.markdown("""
<script>
function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// Scroll immediately and after a short delay
scrollToBottom();
setTimeout(scrollToBottom, 100);

// Also scroll on window resize
window.addEventListener('resize', scrollToBottom);
</script>
""", unsafe_allow_html=True)