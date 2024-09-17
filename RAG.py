import pdfplumber
import faiss
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import dotenv

def get_pdf_text(pdf_doc):
    text=""
    pdf_reader= PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text+= page.extract_text()
    return text

# Example usage:
#ncert_text = get_pdf_text("iesc111.pdf")
#print(ncert_text[:1000])

def initialize_embeddings(text):
    # Initialize embedding model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # Split the text into smaller chunks for embedding
    text_chunks = text.split('\n\n')  # Chunk based on paragraphs or any delimiter
    embeddings = model.encode(text_chunks)
    return model, text_chunks, embeddings

# Example usage:
#model, ncert_text_chunks, embeddings = initialize_embeddings(ncert_text)

def initialize_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Number of dimensions in the embedding
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))  # Adding embeddings to FAISS
    return index

# Example usage:
#index = initialize_faiss_index(embeddings)

def search_similar_text(query, model, index, ncert_text_chunks, k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    similar_text = [ncert_text_chunks[i] for i in I[0]]
    return similar_text

def generate_response(query, documents):
    
    # Load the API key from the .env file
    dotenv.load_dotenv()

    openai = OpenAI(
        api_key=os.getenv("api_key"),
        base_url="https://api.deepinfra.com/v1/openai",
        )
    
    # Create context from retrieved documents
    context = "\n".join(documents)

    # Prepare the system and user messages for the LLaMA model
    messages = [
        {"role": "system", "content": "You are an AI assistant that provides information from NCERT books."},
        {"role": "user", "content": f"Answer the following query based on the context: {context}\nQuery: {query}"}
    ]

    # Send request to OpenAI's LLaMA model
    chat_completion = openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct",  # Use your LLaMA 3.1 model version
        messages=messages
    )

    # Extract the response from the chat completion
    response = chat_completion.choices[0].message.content

    return response

#Agent logic to decide when to call VectorDB or not
def agent_decision(query: str):
    greet = ["hello", "hi", "hey", "greetings", 'good morning', 'good afternoon', 'good evening']
    if query.lower() in greet:
        return "greet"
    elif "summarize" in query.lower() or "overview" in query.lower() or "summary" in query.lower():
        return "summarize"
    else:
        return "vector_db"

# New: Summarize sections of the PDF
def summarize_text(text):
    # Using a simple summary logic for demo purposes
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    
    # Return the summarized text
    return summary[0]['summary_text']

# Initialize FastAPI
app = FastAPI()

# Define request body
class QueryRequest(BaseModel):
    query: str

# Define endpoint for the RAG system
@app.post("/rag")
async def rag_system(query: str = Form(...), pdf_file: UploadFile = File(...)):
    # Step 1: Extract text from PDF
    try:
        pdf_text = get_pdf_text(pdf_file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error reading PDF file")
    
    # Step 2: Initialize embeddings and FAISS index
    model, ncert_text_chunks, embeddings = initialize_embeddings(pdf_text)
    faiss_index = initialize_faiss_index(embeddings)
    
    # Step 3: Retrieve similar documents using FAISS
    similar_documents = search_similar_text(query, model, faiss_index, ncert_text_chunks)
    
    # Step 4: Generate response using OpenAI LLaMA 3.1
    response = generate_response(query, similar_documents)
    
    return {"response": response, "similar_documents": similar_documents}

# Define endpoint for the RAG system with agent logic
@app.post('/agent')
async def agent_system(query : str = Form(...), pdf_file : UploadFile = File(...)):
    # Step 1: Extract text from PDF
    try:
        pdf_text = get_pdf_text(pdf_file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error reading PDF file")
    
    # Step 2: Initialize embeddings and FAISS index
    model, ncert_text_chunks, embeddings = initialize_embeddings(pdf_text)
    faiss_index = initialize_faiss_index(embeddings)
    
    # Step 3: Agent decision
    decision = agent_decision(query)
    
    # Step 4: Perform actions based on agent decision
    if decision == "greet":
        #No need to call the vectorDB
        response = "Hello! How can I help you with the NCERT pdf today?"
        return {"response": response}

    elif decision == "summarize":
        # Step 4: Retrieve similar documents using FAISS
        similar_documents = search_similar_text(query, model, faiss_index, ncert_text_chunks)
        
        # Step 5: Generate response using OpenAI LLaMA 3.1
        response = generate_response(query, similar_documents)

        # Step 6: Summarize the LLM response
        summary = summarize_text(response)
        return {"response": summary}
    
    else:
        # Step 5: Retrieve similar documents using FAISS
        similar_documents = search_similar_text(query, model, faiss_index, ncert_text_chunks)
        
        # Step 6: Generate response using OpenAI LLaMA 3.1
        response = generate_response(query, similar_documents)
    
        return {"response": response}
