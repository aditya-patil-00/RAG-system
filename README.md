<h1 align="center" id="title">RAG System for NCERT PDFs</h1>

<p id="description">This project is a web application built using Streamlit that integrates with FastAPI endpoints for querying and converting text responses into speech. It supports querying NCERT PDFs and provides text-to-speech functionality. The RAG system uses the FAISS vector db for storing the generated text embeddings and is powered with the 'Meta-Llama-3.1-405B-Instruct' model.</p>

<h2> Sample Screenshot </h2>

![Streamlit Interface](https://i.imgur.com/rEpnWop.png)
  
<h2>Features</h2>

Here're some of the project's best features:

*   Query Processing : Query an uploaded NCERT PDF using the RAG or Agent endpoints.
*   Text-to-Speech : Convert the response text to speech and play it directly in the app.

<h2>Installation Steps:</h2>

<p>1. Clone the Repository</p>

```
git clone https://github.com/aditya-patil-00/RAG-system.git
```

<p>2. Create Virtual Env</p>

```
python -m venv myenv
```

<p>3. Install Dependencies</p>

```
pip install -r requirements.txt
```

<p>4. Run FastAPI Backend</p>

```
uvicorn RAG:app --reload
```

<p>5. Run Streamlit App</p>

```
streamlit run app.py
```
