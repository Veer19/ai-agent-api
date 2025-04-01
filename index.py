from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import base64
import tempfile

# Import functions from ai_utils
from ai_utils import (
    load_documents, 
    initialize_vectorizer, 
    find_relevant_documents, 
    create_context, 
    run_agent, 
    create_agents, 
    refine_question
)

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Use the API key in your OpenAI client initialization
client = OpenAI(api_key=openai_api_key)

# Create FastAPI app
app = FastAPI(title="FAQ Agent API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load documents and initialize vectorizer
documents, document_texts = load_documents()
vectorizer, document_vectors = initialize_vectorizer(document_texts)

# Create agents
question_refiner_agent, faq_agent = create_agents()

class Conversation(BaseModel):
    question: str
    answer: str

class AudioRequest(BaseModel):
    audio_data: str  # Base64 encoded audio data
    past_conversations: list[Conversation]

@app.post("/ask-audio")
async def ask_audio_question(request: AudioRequest):
    # Decode the base64 audio data
    audio_bytes = base64.b64decode(request.audio_data)
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
        temp_audio.write(audio_bytes)
        temp_audio.flush()
    
    try:
        # Use OpenAI API for transcription
        with open(temp_audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            
        transcribed_text = transcription.text
        
        # Find relevant documents
        relevant_docs = find_relevant_documents(
            transcribed_text, 
            vectorizer, 
            document_vectors, 
            documents
        )
        
        # Create context with relevant documents and past conversations
        context = create_context(relevant_docs, request.past_conversations)

        # Refine the question based on past conversations
        refined_question = await refine_question(
            question_refiner_agent, 
            transcribed_text, 
            request.past_conversations
        )
        
        # Run the agent with the context and refined question
        answer = await run_agent(faq_agent, context, refined_question)
        
        # If no answer found, retry with full documents
        if "I'm not sure about that" in answer:
            print("Retrying with no chunking")
            relevant_docs = find_relevant_documents(
                refined_question, 
                vectorizer, 
                document_vectors, 
                documents, 
                create_chunks=False
            )
            context = create_context(relevant_docs, request.past_conversations)
            answer = await run_agent(faq_agent, context, refined_question)

        return {"question": refined_question, "answer": answer}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

@app.get("/")
async def root():
    return {"message": "Welcome to the FAQ Agent API. Use /ask-audio endpoint to ask questions."}

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)