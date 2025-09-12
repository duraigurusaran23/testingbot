from dotenv import load_dotenv
import asyncio
import nest_asyncio
import os
import time
from datetime import datetime
import streamlit as st
from chunk_loader import load_processed_chunks, get_chunk_statistics
from pdf_loader import load_pdfs  # Keep as fallback
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from scraper import WebsiteScraper
import shutil
from pathlib import Path
import subprocess



try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

nest_asyncio.apply()
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def delete_existing_data():
    pdfs_dir = Path("../pdfs")
    processed_dir = Path("../processed_data")
    if pdfs_dir.exists():
        shutil.rmtree(pdfs_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    pdfs_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)




st.set_page_config(page_title="Website Knowledge Chatbot", page_icon="💬", layout="wide")

# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = False
if "current_url" not in st.session_state:
    st.session_state.current_url = None
if "scraping" not in st.session_state:
    st.session_state.scraping = False

# Header
col_logo, col_title, col_actions = st.columns([0.1, 0.7, 0.2])
with col_logo:
    st.markdown("### 💬")
with col_title:
    st.markdown("### Website Knowledge Chatbot")
    if st.session_state.current_url:
        st.caption(f"Current site: {st.session_state.current_url}")
    else:
        st.caption("Enter 'new + URL' to scrape a website, or ask questions.")
with col_actions:
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Handle scraping process
if st.session_state.get("scraping", False):
    st.session_state.stop_scraping = False
    progress_bar = st.progress(0.01)
    status_text = st.empty()
    stop_button_placeholder = st.empty()

    def progress_callback(p, s):
        progress_bar.progress(max(p / 100, 0.01))
        status_text.text(s)

    def stop_check():
        return st.session_state.get("stop_scraping", False)

    url = st.session_state.scraping_url
    delete_existing_data()

    # Use direct chunking for faster processing
    scraper = WebsiteScraper()
    chunks = scraper.scrape_to_chunks(url, progress_callback, stop_check)

    if stop_check():
        st.warning("Scraping stopped. Processing partial data...")

    # Save chunks directly without PDF processing
    from datetime import datetime
    import json
    import pickle
    from pathlib import Path

    processed_dir = Path("../processed_data")
    processed_dir.mkdir(exist_ok=True)

    metadata = {
        "processing_date": datetime.now().isoformat(),
        "source_url": url,
        "total_chunks": len(chunks),
        "total_characters": sum(len(chunk) for chunk in chunks)
    }

    # Save chunks
    chunks_file = processed_dir / "chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump({
            "chunks": chunks,
            "metadata": metadata
        }, f, indent=2, ensure_ascii=False)

    pickle_file = processed_dir / "chunks.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump({
            "chunks": chunks,
            "metadata": metadata
        }, f)

    # Clear cache to ensure fresh data loading
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.processed = True
    st.session_state.current_url = url
    st.session_state.scraping = False
    add_message("bot", f"✅ Successfully scraped and processed {url}. You can now ask questions about this website!")
    st.rerun()

model_option = "gemini-1.5-flash"




@st.cache_data(show_spinner=False)
def get_docs():
    """Load pre-processed chunks"""
    status_placeholder = st.empty()

    try:
        status_placeholder.info("📖 Loading pre-processed chunks...")
        docs = load_processed_chunks()

        if docs:
            status_placeholder.success("Bot is ready")
            return docs
        else:
            status_placeholder.warning("⚠️ No pre-processed chunks found. Please scrape a website first by typing 'new + URL'")
            return []

    except Exception as e:
        status_placeholder.error(f"❌ Error loading chunks: {str(e)}")
        return []

docs = get_docs()
global_chunks = docs



@st.cache_resource
def get_chain(selected_model):
    if not docs:
        return None

    try:
        embed_progress = st.empty()
        embed_status = st.empty()

        embed_progress.progress(0.1)
        embed_status.text("🔄 Creating embeddings…")

        # Always try local embeddings first as primary method
        embed_status.text("🔄 Using local embeddings (more reliable)…")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            embedding_provider = "Local (Sentence Transformers)"
            embed_status.text("✅ Local embeddings loaded successfully!")
        except Exception as local_e:
            embed_status.error(f"❌ Failed to load local embeddings: {str(local_e)}")
            embed_status.text("🔄 Trying Google API as fallback…")
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=GOOGLE_API_KEY
                )
                embedding_provider = "Google Gemini"
                embed_status.text("✅ Google API embeddings loaded!")
            except Exception as google_e:
                embed_status.error(f"❌ Both embedding methods failed. Google API Error: {str(google_e)}")
                raise google_e

        embed_progress.progress(0.3)
        embed_status.text("🔄 Building vector store…")

        vector_store = FAISS.from_texts(docs, embeddings)

        embed_progress.progress(0.7)
        embed_status.text("🔄 Configuring retriever…")

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Reduced from 10 for faster retrieval
        )

        embed_progress.progress(0.9)
        embed_status.text("🔄 Initializing chatbot…")

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        custom_prompt_template = """
            You are a professional AI assistant that answers questions based solely on the scraped website content.
            Follow these rules strictly:

            1. ANSWER ONLY FROM CONTEXT:
            - Use only the provided context to answer questions.
            - If information is not in the context, respond: "I don't know. The information is not available in the provided website."
            - Never make assumptions, guess, or use external knowledge.

            2. NATURAL LANGUAGE UNDERSTANDING:
            - Recognize synonyms and related terms:
              * Leadership: CEO, Founder, Owner, President, Director, Executive, Head, Boss, Manager
              * Team/People: Employees, Staff, Workers, Team members, Personnel, Associates
              * Company: Organization, Firm, Business, Corporation, Enterprise
              * Products/Services: Solutions, Offerings, Features, Capabilities
              * Contact: Reach out, Get in touch, Connect, Email, Phone
            - Understand context and intent behind questions

            3. CHAT HISTORY CONTEXT:
            - Use previous conversation to understand follow-up questions
            - If a question is vague, refer to the most recent relevant topic
            - Maintain conversation continuity

            4. RESPONSE STYLE:
            - Be concise but complete
            - Stay focused on the question
            - Use professional, business-appropriate language
            - If question has multiple interpretations, clarify using context/history

            Context: {context}
            Chat History: {chat_history}
            Question: {question}

            Answer:
            """


        CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_prompt_template)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGoogleGenerativeAI(
                model=selected_model,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.0,  # More focused and faster responses
                max_tokens=300,  # Shorter responses for speed
                max_retries=1
            ),
            retriever=retriever,
            memory=memory,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT}
        )

        embed_progress.progress(1.0)
        embed_status.success("✅ Chatbot ready!")
        time.sleep(0.2)
        embed_progress.empty()
        embed_status.empty()

        # Store embedding provider in session state
        st.session_state.embedding_provider = embedding_provider

        return qa_chain
    except Exception as e:
        st.error(f"Failed to initialize the chatbot: {str(e)}")
        if "ResourceExhausted" in str(e) or "quota" in str(e).lower():
            st.error("**API Quota Issue**: The embedding creation failed due to API limits. Try using a different API key or upgrading your Google AI plan.")
        return None

qa_chain = get_chain(model_option)
if qa_chain is None:
    if docs:
        st.error("Failed to initialize embeddings. This may be due to API quota limits. The app will use local embeddings if available.")
        st.error("Please check your API keys or try again later.")
    else:
        st.error("No documents found. Please ensure PDFs are in the 'pdfs' directory and contain text.")
    st.stop()



if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



with st.sidebar:
    st.markdown("## 📊 Data Status")
    if st.session_state.processed:
        stats = {}
        try:
            stats = get_chunk_statistics() or {}
        except Exception:
            stats = {}

        st.metric("Chunks", value=stats.get("total_chunks", "—"))
        st.metric("Characters", value=f"{stats.get('total_characters', 0):,}")
        if st.session_state.current_url:
            st.write(f"**Source:** {st.session_state.current_url}")
    else:
        st.info("No website scraped yet. Type 'new + URL' to get started.")

    st.divider()
    st.markdown("## ⚙️ Model")
    st.write("**Provider:** Google Gemini")
    st.write(f"**Model:** `{model_option}`")
    st.caption("Optimized for factual, contextual answers.")

    st.divider()
    st.markdown("## 🔍 Embeddings")
    if 'embedding_provider' in st.session_state:
        st.write(f"**Provider:** {st.session_state.embedding_provider}")
    else:
        st.write("**Provider:** Local (Fast)")

    st.divider()
    st.markdown("### Usage Tips")
    st.caption("• Type 'new + URL' to scrape a website")
    st.caption("• Ask questions about the scraped content")
    st.caption("• Answers are based only on website data")



def now_str():
    return datetime.now().strftime("%H:%M")

def add_message(role, text):
    st.session_state.messages.append({
        "role": role,          
        "text": text,
        "ts": now_str()
    })

def handle_user_query(user_input):
    """Processes user input, gets a response, updates state."""
    if not user_input.strip():
        return

    add_message("user", user_input)

    # Check if user wants to scrape a new website
    if user_input.lower().startswith("new "):
        url_part = user_input[4:].strip()
        if url_part.startswith("http"):
            # Start scraping process
            st.session_state.scraping = True
            st.session_state.scraping_url = url_part
            add_message("bot", f"🔄 Starting to scrape {url_part}... Please wait.")
            st.rerun()
        else:
            add_message("bot", "Please provide a valid URL starting with http:// or https://")
        return

    # If no data is processed yet, prompt user to scrape a site
    if not st.session_state.processed:
        add_message("bot", "Please scrape a website first by typing 'new + URL' (e.g., 'new https://example.com')")
        return

    with st.spinner("🔍 Searching documents…"):
        try:
            time.sleep(0.1)  # Reduced delay for faster response

            with st.spinner("🤖 Generating response…"):
                result = qa_chain.invoke({
                    "question": user_input,
                    "chat_history": st.session_state.get("chat_history", [])
                })
                answer = result.get("answer", "I couldn't generate a response.")

            add_message("bot", answer)
            st.session_state.chat_history = st.session_state.get("chat_history", []) + [(user_input, answer)]
        except Exception as e:
            error_msg = "Sorry, I encountered an error while processing your request."
            if "ResourceExhausted" in str(e) or "quota" in str(e).lower():
                error_msg = "⚠️ **API Quota Exceeded**: You've hit the API limit. Please wait or upgrade your plan."
            else:
                error_msg += f"\n\nDetails: {str(e)}"
            add_message("bot", error_msg)



st.markdown("""
    <style>
        /* Global */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        /* Main container */
        .main .block-container {
            padding: 1rem 2rem;
            max-width: 1000px;
        }

        /* Header styling */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #ffffff;
            font-weight: 600;
        }

        /* Chat shell */
        .chat-shell {
            max-width: 900px;
            margin: 0 auto;
            border-radius: 20px;
            background: #FFFFFF;
            border: 1px solid #E6E8ED;
            display: flex;
            flex-direction: column;
            height: calc(100vh - 180px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
            overflow: hidden;
        }

        /* Top bar */
        .chat-topbar {
            padding: 16px 20px;
            border-bottom: 1px solid #E9EEF5;
            display: flex; align-items: center; justify-content: space-between;
            position: sticky; top: 0; background: #FFFFFF; z-index: 10;
            border-top-left-radius: 20px; border-top-right-radius: 20px;
        }
        .chat-title {
            font-weight: 700;
            color: #1f2937;
            font-size: 1.1rem;
        }

        /* Scrollable history */
        .chat-history {
            flex: 1;
            overflow-y: auto;
            padding: 20px 18px 10px 18px;
            display: flex; flex-direction: column; gap: 12px;
            scroll-behavior: smooth;
            background: #fafafa;
        }

        /* Message rows */
        .msg-row {
            display: flex;
            gap: 10px;
            align-items: flex-end;
            animation: fadeIn 0.3s ease-in;
        }
        .msg-row.user { justify-content: flex-end; }
        .msg-row.bot { justify-content: flex-start; }

        /* Avatars */
        .avatar {
            width: 32px; height: 32px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 14px; background: #EEF2FF; border: 2px solid #E0E7FF;
            color: #3F51B5; flex-shrink: 0;
            font-weight: 600;
        }
        .avatar.user {
            background: linear-gradient(135deg, #10b981, #059669);
            border-color: #047857;
            color: white;
        }

        /* Bubbles */
        .bubble {
            max-width: 75%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.6;
            font-size: 0.95rem;
            word-wrap: break-word; white-space: pre-wrap;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .bot .bubble {
            background: #FFFFFF; color: #374151; border: 1px solid #E5E7EB;
            border-bottom-left-radius: 6px;
        }
        .user .bubble {
            background: linear-gradient(135deg, #10b981, #059669); color: #FFFFFF;
            border-bottom-right-radius: 6px;
        }

        /* Timestamp */
        .ts {
            font-size: 0.75rem; color: #9CA3AF; margin: 0 8px;
            font-weight: 500;
        }

        /* Bottom input bar */
        .input-bar {
            border-top: 1px solid #E9EEF5;
            padding: 12px 18px;
            background: #FFFFFF;
            position: sticky; bottom: 0; z-index: 10;
            border-bottom-left-radius: 20px; border-bottom-right-radius: 20px;
        }
        .input-bar .stTextInput > div > div {
            border-radius: 25px !important;
            border: 2px solid #E5E7EB !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
            transition: all 0.2s ease;
        }
        .input-bar .stTextInput > div > div:focus-within {
            border-color: #10b981 !important;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
        }
        .send-row .stButton>button {
            height: 44px; border-radius: 22px;
            border: 2px solid #10b981;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .send-row .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }

        /* Scrollbar */
        .chat-history::-webkit-scrollbar { width: 6px; }
        .chat-history::-webkit-scrollbar-thumb {
            background: #CBD5E1; border-radius: 3px;
        }
        .chat-history::-webkit-scrollbar-thumb:hover {
            background: #94A3B8;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Sidebar improvements */
        .stSidebar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
        }
    </style>
    """, unsafe_allow_html=True)





st.markdown('<div id="chat-history" class="chat-history">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    role = msg.get("role", "bot")
    text = msg.get("text", "")
    ts = msg.get("ts", now_str())
    is_user = role == "user"
    avatar = "🙋" if is_user else "🤖"
    row_class = "user" if is_user else "bot"
    st.markdown(
        f"""
        <div class="msg-row {row_class}">
            {'<div class="avatar user">🙋</div>' if is_user else '<div class="avatar">🤖</div>'}
            <div class="bubble">{text}</div>
            <div class="ts">{ts}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)



st.markdown('<div class="input-bar">', unsafe_allow_html=True)
with st.form("chat-input-form", clear_on_submit=True):
    input_cols = st.columns([1, 0.12])
    user_text = input_cols[0].text_input(
        "Ask a question:",
        key="user_input_box",
        placeholder="Type your message…",
        label_visibility="collapsed",
    )
    submitted = input_cols[1].form_submit_button("Send ➤")
    if submitted and user_text:
        handle_user_query(user_text)
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <script>
        const hist = window.parent.document.querySelector('#chat-history');
        if (hist) { hist.scrollTop = hist.scrollHeight; }
    </script>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
