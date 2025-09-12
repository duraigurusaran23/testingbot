# 🚀 RAG (Retrieval-Augmented Generation) Chatbot

This project implements a powerful AI chatbot that can answer questions about your own PDF documents. It uses Google's Gemini models, LangChain for orchestration, and Streamlit for the user interface.

## ✨ Key Features

-   **Pre-processing Pipeline**: Process your PDFs once for ultra-fast startup times.
-   **Vector Search**: Uses `FAISS` for efficient similarity search to find relevant information.
-   **Flexible Models**: Easily switch between different Gemini models (`2.0 Flash`, `1.5 Pro`, etc.).
-   **Web Interface**: A clean, user-friendly chat interface powered by Streamlit.
-   **Smart Caching**: Caches processed data and embeddings to avoid redundant work.
-   **Debugging Tools**: Extensive logging to diagnose retrieval and generation issues.

## 📁 Project Structure

```
rag-Project/
├── .venv/                    # Python virtual environment
├── pdfs/                     # Place your PDF documents here
│   └── your_company_data.pdf
├── backend/                  # Main application source code
│   ├── app.py               # Streamlit chatbot UI and logic
│   ├── preprocess.py        # One-time PDF chunking script
│   ├── chunk_loader.py      # Loads pre-processed chunks
│   ├── pdf_loader.py        # PDF text extraction (fallback)
│   ├── .env                 # API keys and environment variables
│   └── requirements.txt     # Python package dependencies
└── processed_data/          # Auto-generated chunk storage (for speed)
    ├── chunks.pkl          # Fast-loading binary format
    └── chunks.json         # Human-readable format for debugging
```

---

## 🎯 **Step-by-Step: How to Run This Project**

### **Step 1: Setup the Environment**

1.  **Navigate to the project directory:**
    ```bash
    cd c:/Users/dgskt/OneDrive/Desktop/rag-Project
    ```
2.  **Activate the Python virtual environment:**
    ```bash
    # For Windows Command Prompt
    call .venv\Scripts\activate.bat
    ```
    *(Your command prompt should now be prefixed with `(.venv)`)*
3.  **Install dependencies** (if you haven't already):
    ```bash
    pip install -r backend/requirements.txt
    ```

### **Step 2: Configure Your API Key**

1.  Open the file `backend/.env`.
2.  Add your Google Gemini API key:
    ```
    GOOGLE_API_KEY=AIzaSy...your...key...here
    ```

### **Step 3: Add Your Documents**

1.  Place any PDF files you want the chatbot to read inside the `pdfs/` directory.

### ⭐ **Step 4: Preprocess Your PDFs (Crucial First Step!)**

This step reads your PDFs, breaks them into chunks, and saves them for fast loading. You only need to run this once, or whenever you add/change your PDFs.

```bash
# Navigate to the backend directory
cd backend

# Run the preprocessing script
python preprocess.py
```

You should see output confirming that the PDFs were found, processed, and the chunks were saved.

### **Step 5: Run the Chatbot!**

Now you can start the web application.

```bash
# Make sure you are still in the 'backend' directory
streamlit run app.py
```

### **Step 6: Use the Chatbot**

1.  Your browser should automatically open to the app's URL (usually `http://localhost:8501`).
2.  Use the sidebar to select your preferred Gemini model.
3.  Start asking questions about the content of your PDFs!

---

## 🔧 **Advanced Usage & Troubleshooting**

### **When to Reprocess PDFs**

Run the `preprocess.py` script again if you:
-   Add new PDFs to the `pdfs/` folder.
-   Modify existing PDFs.
-   Change the chunking settings in `preprocess.py`.

### **Clearing the Cache**

If the app feels "stuck" or isn't reflecting changes, use the **"🔄 Clear Cache & Restart"** button in the app's sidebar. This clears Streamlit's cache and reloads the embeddings.

### **Debugging Retrieval**

If the chatbot can't find information that you know is in the PDF:
1.  Check the terminal output where you launched Streamlit.
2.  The `DEBUG` messages will show:
    -   If the correct chunks were loaded.
    -   How many documents were retrieved for your query.
    -   A preview of the retrieved document content.
3.  If "Retrieved 0 documents" appears, it means the vector search is failing. Try adjusting the `search_kwargs` in `app.py` or the chunking strategy in `preprocess.py`.

### **API Quota Errors**

If you see a `ResourceExhausted` error:
-   Try switching to a model with a higher free-tier limit, like `gemini-1.5-flash`, using the sidebar.
-   Wait a few minutes for your API quota to reset.
-   Consider upgrading your Google AI plan.