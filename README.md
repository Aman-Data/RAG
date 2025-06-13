﻿# 🤖 RAG PDF Q\&A App with Streamlit

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** application built using **Streamlit**, **LangChain**, **FAISS**, and **Groq API**. It allows users to upload a PDF, convert it into vector embeddings using `sentence-transformers`, and then ask questions about the content. The app retrieves relevant document chunks and uses an LLM (e.g., `gemma2-9b-it`) to generate accurate answers based only on the given context.

---

## 📌 Features

* 📄 Read and process PDFs with `PyPDF2`
* ✂️ Split large documents into smaller chunks using `LangChain`
* 🔍 Create semantic search over documents using `FAISS` and HuggingFace Embeddings
* 🧠 Use LLMs from **Groq** for fast and cost-effective generation
* 🧾 Retrieval-based answers with no hallucination
* 🧑‍💻 Streamlit-based interactive UI

---

## 🚀 Tech Stack

* **Python 3.10+**
* [Streamlit](https://streamlit.io/)
* [LangChain](https://www.langchain.com/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Hugging Face Transformers](https://huggingface.co/)
* [Groq API](https://console.groq.com/)
* **Model Used**: `gemma2-9b-it` (via Groq)

---

## 📁 Project Structure

```
rag-pdf-app/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/rag-pdf-app.git
cd rag-pdf-app
```

2. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install Requirements**

```bash
pip install -r requirements.txt
```

> If `faiss` gives issues on Windows, use `faiss-cpu`:

```bash
pip install faiss-cpu
```

---

## 📥 Requirements File

`requirements.txt`

```
streamlit
PyPDF2
langchain
langchain-community
sentence-transformers
faiss-cpu
```

---

## 🔐 Setting Up GROQ API

Create a free account at [https://console.groq.com](https://console.groq.com) and get your API key.

In `app.py`, replace:

```python
groqapi = ''
```

with:

```python
groqapi = 'your-groq-api-key-here'
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Upload PDF**: The file is read using PyPDF2.
2. **Split Text**: It’s chunked into manageable parts using recursive text splitting.
3. **Vectorization**: Each chunk is embedded using HuggingFace transformer models.
4. **Indexing**: FAISS creates an efficient similarity search index.
5. **Ask Questions**: The question is matched with the most relevant document chunks.
6. **LLM Response**: The context is passed to a Groq-powered LLM which generates the answer.

---

## 📝 Prompt Template Used

```text
You are a helpful assistant. Answer the question using only the context below.
If the answer is not present, just say no. Do not try to make up an answer.

Context:
{context}

Question:
{question}

Helpful Answer:
```

---

## 🧪 Example Use Case

* Upload a PDF titled "What is AI?"
* Ask: *"What is the purpose of artificial intelligence?"*
* The app returns an answer based only on the PDF content.

---

## 📎 To-Do / Improvements

* [ ] Add support for multiple files
* [ ] Add memory/chat history
* [ ] Deploy on Streamlit Cloud or Hugging Face Spaces
* [ ] Upload PDFs through UI instead of hardcoding file path
* [ ] Add file type validations

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Aman Tadvi**

