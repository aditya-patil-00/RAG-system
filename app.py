import streamlit as st
import requests
import base64
import io

# Define FastAPI backend URLs
RAG_API_URL = "http://127.0.0.1:8000/rag"  # RAG endpoint URL
AGENT_API_URL = "http://127.0.0.1:8000/agent"  # Agent endpoint URL
TTS_API_URL = "https://api.sarvam.ai/text-to-speech" # Text-to-Speech endpoint URL  

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
                
            # Convert response to speech
            #st.button("Convert Response to Speech")
                audio_data = text_to_speech(response["response"])
                if audio_data:
                    st.write("Audio generated successfully. You can play it below:")
                    st.audio(audio_data, format="audio/wav")
                else:
                    st.error("Failed to generate audio.")
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
    
def text_to_speech(text):
    url = TTS_API_URL
    
    payload = {
        "inputs": [text],
        "target_language_code": "hi-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.65,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    headers = {
        "api-subscription-key": "7b9282e5-7bc3-4e5f-b1fd-eb925d10b2e7",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'audios' in data and data['audios']:
                audio_base64 = data['audios'][0]
                # Decode base64 audio
                audio_data = base64.b64decode(audio_base64)
                return io.BytesIO(audio_data)
            else:
                st.error("Unexpected response format.")
                return None
        else:
            st.error(f"Failed to generate speech: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error while processing text-to-speech request: {e}")
        return None

if __name__ == "__main__":
    main()