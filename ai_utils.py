import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from agents import Agent, Runner
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

# Load documents for similarity search
def load_documents(docs_directory="documents"):
    documents = []
    
    # Load all txt files from the documents directory
    if os.path.exists(docs_directory):
        for filename in os.listdir(docs_directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(docs_directory, filename)
                with open(file_path, "r") as f:
                    content = f.read()
                    documents.append({"title": filename, "content": content})
    
    return documents




def find_relevant_documents(
    query: str,
    documents: List[Dict[str, str]],
    chunk: bool = True,
    top_n_docs: int = 3,
    chunk_size: int = 3,
    similarity_threshold: float = 0.4
) -> List[Dict[str, str]]:
    """
    Returns relevant document chunks (or whole documents) using semantic similarity.
    
    Parameters:
    - query: User question.
    - documents: List of documents (each with 'title' and 'content').
    - chunk: Whether to dynamically chunk documents.
    - top_n_docs: How many top documents to consider for chunking.
    - chunk_size: Number of paragraphs per chunk (if chunking is enabled).
    - similarity_threshold: Minimum cosine similarity score to include a result.
    """
    if not documents:
        return []

    # Load model (caches after first use)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Extract texts and titles
    doc_texts = [doc["content"] for doc in documents]
    doc_titles = [doc["title"] for doc in documents]

    # Step 1: Embed full documents
    doc_vectors = model.encode(doc_texts, convert_to_tensor=True)

    # Step 2: Embed the query
    query_vector = model.encode(query, convert_to_tensor=True)

    # Step 3: Find top-N relevant documents
    doc_similarities = util.pytorch_cos_sim(query_vector, doc_vectors)[0]
    top_doc_indices = doc_similarities.argsort(descending=True)[:top_n_docs]

    relevant_results = []

    for doc_idx in top_doc_indices:
        full_text = doc_texts[doc_idx]
        title = doc_titles[doc_idx]
        doc_score = doc_similarities[doc_idx].item()

        if not chunk:
            if doc_score >= similarity_threshold:
                relevant_results.append({
                    "title": title,
                    "content": full_text.strip(),
                    "score": round(doc_score, 3)
                })
            continue

        # If chunking is enabled
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        chunked_texts = [
            "\n\n".join(paragraphs[i:i+chunk_size])
            for i in range(0, len(paragraphs), chunk_size)
        ]

        chunk_vectors = model.encode(chunked_texts, convert_to_tensor=True)
        chunk_similarities = util.pytorch_cos_sim(query_vector, chunk_vectors)[0]

        best_idx = chunk_similarities.argmax().item()
        best_score = chunk_similarities[best_idx].item()

        if best_score >= similarity_threshold:
            relevant_results.append({
                "title": title,
                "content": chunked_texts[best_idx],
                "score": round(best_score, 3)
            })

    return sorted(relevant_results, key=lambda x: x["score"], reverse=True)


def create_context(relevant_docs):
    print("--------------------------------")
    print("Creating context")
    print("Relevant docs:")
    print(relevant_docs)
    print("--------------------------------")
    # Create context with relevant documents
    context = ""
    # Add past questions to the context if available
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
            If there are no past conversations, just return the question as it is.
            
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

            1. Use clear, casual, and human-like phrasing.

            2. Avoid robotic or overly formal language.

            3. Convert lists into scentences.

            Input Format:
            Context: List of all relevant documents to help answer the question.
            Question: The question asked by the user.

            Answer the question using the context.

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