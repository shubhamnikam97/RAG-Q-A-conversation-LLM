# 📚 Conversational RAG Chatbot with PDF Upload + Chat History  
A Streamlit-based **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload PDFs and interact with their content conversationally.  
The system maintains full **chat history**, reformulates contextual questions, and retrieves relevant document chunks using embeddings and a vector database.

Built using **LangChain**, **Groq LLM**, **HuggingFace Embeddings**, and **ChromaDB**.

---

## 🚀 Features

- **Conversational PDF Q&A**  
  Ask natural-language questions about your PDFs using AI.

- **Chat History Memory**  
  Each session stores its own conversation context using LangChain's message history.

- **RAG Pipeline**  
  Uses:
  - Document splitting  
  - Embeddings  
  - Vector storage  
  - Context-aware question rewriting  
  - Document retrieval  
  - LLM answer generation  

- **Multiple PDF Support**  
  Upload one or more PDFs per session.

- **Local Embeddings (Fast & Free)**  
  Uses `all-MiniLM-L6-v2` via HuggingFace to create embeddings locally.

- **Groq LLM Integration**  
  Powered by the ultra-fast `llama-3.1-8b-instant` model.

- **History-Aware Retrieval**  
  The system automatically rephrases questions using chat context before querying the vector store.

---

## 🧠 How the System Works

This RAG application follows a multi-step pipeline:

### **1️⃣ PDF Upload & Processing**
- PDFs are uploaded through Streamlit
- Processed using **PyPDFium2Loader**
- Split into overlapping chunks via **RecursiveCharacterTextSplitter**

### **2️⃣ Embeddings + Vector DB**
- Chunks are converted into dense embeddings using:
- Stored inside a **Chroma** vector database

### **3️⃣ History-Aware Retrieval**
Before retrieving, the system asks the LLM to rewrite the user’s question using prior messages:

> *"Given chat history + question, generate a standalone version of this question."*

This improves retrieval accuracy.

### **4️⃣ RAG Chain**
Retrieved documents are passed into a prompt + Groq LLM to generate final concise answers.

### **5️⃣ Chat Memory**
Each session ID maintains persistent chat history using:


This allows long conversations about PDFs.

---

## 📦 Tech Stack

- **Python 3.12+**
- **Streamlit**
- **LangChain Classic**
- **LangChain Chroma**
- **HuggingFace Embeddings**
- **Chroma Vector DB**
- **Groq LLM API**
- **PyPDFium2 + PyPDFLoader**

---

## 📁 Project Structure
```
.
├── app.py # Main RAG Chatbot application
├── requirements.txt # Dependencies
└── README.md # Documentation
```
---

## 🔧 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

2️⃣ Create & activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Add HuggingFace & Groq API keys
```bash
HF_TOKEN=your_hf_token
GROQ_API_KEY=your_groq_api_key
```

▶️ Run the Application
```bash
streamlit run "file_path/app.py"
```

🖥 Usage Instructions
✔ Step 1 — Enter your Groq API key

Using sidebar or .env.

✔ Step 2 — Upload PDFs

Supports multiple documents.

✔ Step 3 — Enter a session ID

Example: my_research_session

✔ Step 4 — Ask questions

Examples:

“Summarize section 2.”

“What did the author say about neural networks?”

“How does this compare to chapter 3?”

✔ Step 5 — View chat history

The app displays real-time session memory.
