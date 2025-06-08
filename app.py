import streamlit as st
import PyPDF2
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

st.header('A RAG App')

groqapi='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'

from langchain.text_splitter import RecursiveCharacterTextSplitter

uploaded_file= r"C:\Users\amant\Downloads\What is AI.pdf"

#Read PDF with PyPDF2
text=""
pdf_reader=PyPDF2.PdfReader(uploaded_file)
for page in pdf_reader.pages:
    text +=page.extract_text()+ "\n"

#Split text into chunks (strings)
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
text_chunks=splitter.split_text(text)

# Convert each chunk to Document object
docs=[Document(page_content=chunk)for chunk in text_chunks]
st.subheader('Document splitted successfully')

# Create embeddings model
embeddigs=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Create FAISS vector store from documents
vectordb=FAISS.from_documents(docs,embeddigs)

st.success('FAISS Vectorstore created')
retriever=vectordb.as_retriever()

from langchain.chat_models import init_chat_model
model=init_chat_model(model='gemma2-9b-it',model_provider='groq',api_key=groqapi)

from langchain.prompts import PromptTemplate

template="""
You are a helpful assistant. Answer the question using only the context below.
If the answer is not present, just say no. Do not try to make up an answer.

context:
{context}

Question:
{question}

Helpful Answer:
"""

rag_prompt= PromptTemplate(input_variables=['Context','question'],template=template)

user_query=st.text_input('ASk a question about the PDF')

if user_query:
    relevant_docs=retriever.invoke(user_query)

    final_prompt=rag_prompt.format(context=relevant_docs,question=user_query)

    with st.spinner('Generating Answer...'):
        response=model.invoke(final_prompt)

    st.write('Answer')
    st.write(response.content)



