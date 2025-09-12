# Detailed Application Workflow: Step-by-Step Execution

This document provides a granular, chronological explanation of the file interactions and processes that occur after you execute `streamlit run app.py`.

### Pre-computation Step (Manual)

Before the app is ever run, you must manually run `preprocess.py`.

1.  **File Executed**: `backend/preprocess.py`
2.  **Action**:
    *   Scans the `pdfs/` directory.
    *   Opens each PDF, extracts all text.
    *   Uses a `RecursiveCharacterTextSplitter` to break the text into small, overlapping chunks (currently 600 characters each).
    *   Saves these chunks into two files in the `processed_data/` directory: `chunks.json` (for readability) and `chunks.pkl` (for fast loading).
3.  **Result**: A folder (`processed_data/`) now exists, containing the entire content of your PDFs, pre-digested and ready for the main application.

---

## ▶️ Phase 1: Application Initialization

This is the sequence of events when you start the Streamlit application.

### Step 1: `streamlit run app.py` is Executed

*   **File:** `backend/app.py` (Lines 1-28)
*   **What Happens:**
    *   The Python interpreter starts executing `app.py` from top to bottom.
    *   **Imports:** Essential libraries are loaded (`streamlit`, `langchain`, `dotenv`).
    *   **Crucial Import:** `from chunk_loader import load_processed_chunks` makes the function to load our pre-processed data available.
    *   **Environment Variables:** `load_dotenv()` is called, which reads `backend/.env` and loads your `GOOGLE_API_KEY` into the environment.
    *   **Streamlit Setup:** `st.set_page_config()` and `st.title()` set up the basic web page structure and title.

### Step 2: Loading the Document Chunks

*   **File:** `backend/app.py` (Line 104: `@st.cache_data`)
*   **What Happens:**
    *   Streamlit sees the `@st.cache_data` decorator above the `get_docs()` function. This tells Streamlit to run this function once and then save the result. It will not run it again unless the code of the function changes.
    *   **Function Call:** `get_docs()` is executed.
    *   **Calls `chunk_loader.py`:** Inside `get_docs()`, the line `docs = load_processed_chunks()` is called.
    *   **File:** `backend/chunk_loader.py` (Function: `load_chunks`)
        *   This function looks for `processed_data/chunks.pkl`.
        *   It finds the file, opens it, and loads the pre-processed chunks into a Python list.
        *   It returns this list of chunks back to `app.py`.
    *   The `get_docs()` function now returns the list of 11 text chunks.
    *   **Caching:** Streamlit caches this list. For the rest of the session, any time `get_docs()` is called, it will instantly return the cached list without re-reading the file.

### Step 3: Building the QA Chain and Vector Store

*   **File:** `backend/app.py` (Line 139: `@st.cache_resource`)
*   **What Happens:**
    *   Streamlit sees the `@st.cache_resource` decorator above `get_chain()`. This is similar to caching data, but for live objects like database connections or, in our case, the QA Chain.
    *   **Function Call:** `get_chain()` is executed.
    *   **Instantiate Embeddings:** `GoogleGenerativeAIEmbeddings` is created. This object knows how to communicate with the Google API to turn text into vectors.
    *   **Instantiate LLM:** `ChatGoogleGenerativeAI` is created, pointing to the specific Gemini model you selected in the sidebar.
    *   **⭐ CRITICAL STEP: Vector Store Creation**:
        *   `FAISS.from_texts(docs, embeddings)` is called.
        *   For **each of the 11 text chunks**, the `embeddings` object makes an API call to Google, gets a vector representation, and `FAISS` stores it in an efficient, searchable index.
        *   This creates the `vector_store` object. This is the "brain" that knows the content of your documents.
    *   **Create Retriever:** `vector_store.as_retriever()` creates an object specifically designed to search the vector store. The `search_kwargs` configure it to be very permissive (return up to 10 results with a similarity score of at least 0.1).
    *   **Create QA Chain:** `RetrievalQA.from_chain_type()` assembles all the pieces: the LLM and the retriever. It's now a complete object ready to answer questions.
    *   **Caching:** Streamlit caches the entire `qa_chain` object. It will not be rebuilt for the rest of the session.

### Step 4: Final UI Setup

*   **File:** `backend/app.py` (Lines 200+)
*   **What Happens:**
    *   The main body of the Streamlit script runs.
    *   It initializes the session state for messages (`st.session_state.messages`).
    *   It renders the text input box: `st.text_input(...)`.
    *   The application is now fully loaded and is waiting for user input.

---

## 💬 Phase 2: User Asks a Question

This is what happens every time you type a question and press Enter.

### Step 1: User Input is Submitted

*   **File:** `backend/app.py` (Line 211: `on_change=send_message`)
*   **What Happens:**
    *   The `on_change` callback triggers the `send_message()` function.
    *   The user's text is appended to the chat history.

### Step 2: Retrieving Relevant Documents

*   **File:** `backend/app.py` (Inside `send_message()`)
*   **What Happens:**
    *   The line `result = qa_chain.invoke({"query": user_input})` starts the process.
    *   **Inside the QA Chain:**
        *   The `retriever` object takes the `user_input` ("what is the person name").
        *   It makes an API call to Google to get the vector for this question.
        *   It then searches the `FAISS` vector store for the 10 vectors that are most similar to the question's vector.
        *   **THIS IS THE FAILING STEP:** The log `Retrieved 0 documents` shows that this search is finding **no similar vectors**, even with a very low threshold. This is the core of the problem.

### Step 3: Generating the Answer

*   **File:** `backend/app.py` (Inside `send_message()`)
*   **What Happens:**
    *   The QA Chain takes the (empty list of) retrieved documents and combines them with the user's question into a prompt.
    *   It sends this prompt to the selected Gemini LLM.
    *   The LLM receives a prompt like: *"Based on the following context, answer the user's question. Context: []. Question: what is the person name"*.
    *   Since there is no context, the LLM correctly responds that it doesn't have enough information.
    *   The answer `result` is returned.

### Step 4: Displaying the Result

*   **File:** `backend/app.py` (Inside `send_message()`)
*   **What Happens:**
    *   The LLM's response is appended to the chat history.
    *   Streamlit automatically re-renders the web page, showing the new question and the AI's (incorrect) answer.