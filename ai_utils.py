import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from agents import Agent, Runner

# Load documents for similarity search
def load_documents(docs_directory="documents"):
    documents = []
    document_texts = []
    
    # Load all txt files from the documents directory
    if os.path.exists(docs_directory):
        for filename in os.listdir(docs_directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(docs_directory, filename)
                with open(file_path, "r") as f:
                    content = f.read()
                    documents.append({"title": filename, "content": content})
                    document_texts.append(content)
    
    return documents, document_texts

# Initialize TF-IDF vectorizer
def initialize_vectorizer(document_texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    if document_texts:
        document_vectors = vectorizer.fit_transform(document_texts)
        return vectorizer, document_vectors
    return vectorizer, None

# Function to find relevant documents
def find_relevant_documents(query, vectorizer, document_vectors, documents, top_n=1, create_chunks=True):
    if document_vectors is None or len(documents) == 0:
        return []
    
    # Transform query to vector
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity with whole documents first
    similarities = cosine_similarity(query_vector, document_vectors).flatten()
    
    # Get indices of top similar documents
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Return relevant document chunks instead of full documents
    relevant_docs = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Only include if similarity is above threshold
            # Get the document content
            content = documents[idx]['content']
            chunks = []
            
            # For other documents, split by paragraphs or sections
            paragraphs = content.split("\n\n")
           
            for para in paragraphs:
                if para.strip():
                    chunks.append({
                        "title": documents[idx]['title'], 
                        "content": para.strip()
                    })
            
            # Find the most relevant chunk within the document
            if create_chunks and chunks:
                # Extract text from chunks for vectorization
                chunk_texts = [chunk["content"] for chunk in chunks]
                # Transform chunks to vectors
                chunk_vectors = vectorizer.transform(chunk_texts)
                # Calculate similarity between query and each chunk
                chunk_similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
                # Get the most relevant chunk
                best_chunk_idx = chunk_similarities.argmax()
                # Only add the chunk if it's sufficiently relevant
                if chunk_similarities[best_chunk_idx] > 0.1:
                    relevant_docs.append(chunks[best_chunk_idx])
            else:
                # Fallback to the whole document if chunking fails
                relevant_docs.append(documents[idx])
    
    return relevant_docs

def create_context(relevant_docs, past_conversations):
    print("--------------------------------")
    print("Creating context")
    print("Relevant docs:")
    print(relevant_docs)
    print("Past conversations:")
    print(past_conversations)
    print("--------------------------------")
    # Create context with relevant documents
    context = ""
    # Add past questions to the context if available
    context += "Past Conversations:\n"
    for q in past_conversations:
        context += f"User: {q.question}\n"
        context += f"Assistant: {q.answer}\n\n"
    
    if relevant_docs:
        context = "\nContext:\n\n"
        for doc in relevant_docs:
            context += f"--- {doc['title']} ---\n{doc['content']}\n\n"
    return context

async def run_agent(agent, context, question):
    print("Running agent")
    print("Context:")
    print(context)
    print("Question:")
    print(question)
    result = await Runner.run(agent, f"{context}\nQuestion: {question}")
    print("Result:")
    print(result.final_output)
    return result.final_output

# Create agents
def create_agents():
    # Create question refiner agent
    question_refiner_agent = Agent(
        name="Question Refiner",
        instructions="""
            Role:
            You are an AI assistant that specializes in understanding user intent and refining questions.
            
            Task:
            Analyze the user's current question and past conversations to generate a more precise question
            that captures their true intent. Consider context from previous interactions to clarify ambiguities.
            
            Input Format:
            Past Conversations: List of all previous conversations between the user and the assistant.
            Question: The current question asked by the user.
            
            Output Format:
            A refined version of the user's question that better captures their intent.
            If the original question is already clear, return it unchanged.
        """
    )
    
    # Create FAQ agent
    faq_agent = Agent(
        name="FAQ Agent",
        instructions="""
            Role:
            You are a voice assistant designed to answer FAQs using only the provided context. You must speak in a natural, friendly, and conversational tone—like a helpful human, not a script.

            Behavior Rules:

            1. Context-Bound Responses Only

            2. Respond only if the answer exists in the provided context.

            3. Do not guess, infer, or fabricate any information.

            Missing Information Handling

            If the answer is not in the context, say:
            "I'm not sure about that—let me connect you to a support expert who can help!"

            Do not attempt to answer under any circumstances.

            Conversational Style

            4. Use clear, casual, and human-like phrasing.

            5. Avoid robotic or overly formal language.

            Input Format:
            Past Conversations: List of all previous conversations between the user and the assistant.
            Context: List of all relevant documents to help answer the question.
            Question: The question asked by the user.

            Use the past conversations to form a better question, then answer the question using the past answers and the context.

            Output Format:
            Speak in short, friendly sentences.
            Aim for clarity and empathy in every response.
        """,
    )
    
    return question_refiner_agent, faq_agent

async def refine_question(agent, question, past_conversations):
    if not past_conversations:
        return question
        
    context = "Past Conversations:\n"
    for q in past_conversations:
        context += f"User: {q.question}\n"
        context += f"Assistant: {q.answer}\n\n"
    
    result = await Runner.run(agent, f"{context}\nCurrent Question: {question}")
    refined_question = result.final_output
    print(f"Original question: {question}")
    print(f"Refined question: {refined_question}")
    return refined_question 