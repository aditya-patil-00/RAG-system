import streamlit as st
import requests

# Define FastAPI backend URLs
RAG_API_URL = "http://127.0.0.1:8000/rag"  # RAG endpoint URL
AGENT_API_URL = "http://127.0.0.1:8000/agent"  # Agent endpoint URL

def main():
    st.title("NCERT RAG System")

    # Let the user choose between the RAG and Agent endpoints
    endpoint_option = st.selectbox(
        "Select the endpoint you want to query:",
        ("RAG Endpoint", "Agent Endpoint")
    )

    # File uploader for PDF
    pdf_file = st.file_uploader("Upload NCERT PDF", type=["pdf"])

    # Input query
    query = st.text_input("Enter your query:")

    # Submit button to start processing
    if st.button("Submit"):
        if pdf_file is not None and query:
            # Decide which endpoint to use based on the user selection
            if endpoint_option == "RAG Endpoint":
                api_url = RAG_API_URL
            else:
                api_url = AGENT_API_URL

            # Send request to the selected API endpoint
            response = process_query(api_url, query, pdf_file)

            # Display results
            if response:
                st.write("Response from the system:")
                st.write(response["response"])
                st.write("Similar Documents:")
                for doc in response["similar_documents"]:
                    st.write(f"- {doc}")
        else:
            st.error("Please provide both a query and a PDF file.")

def process_query(api_url, query, pdf_file):
    # Define payload for the FastAPI endpoint
    files = {
        'query': (None, query),
        'pdf_file': pdf_file
    }
    
    try:
        # Send POST request to the FastAPI endpoint (either RAG or Agent)
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            return response.json()  # Return the JSON response from the backend
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error while processing request: {e}")
        return None

if __name__ == "__main__":
    main()
