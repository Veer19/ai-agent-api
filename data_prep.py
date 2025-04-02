import os
import json
from pathlib import Path
from typing import List, Dict
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def extract_faqs_from_document(document_content: str, document_title: str) -> List[Dict]:
    """
    Uses OpenAI to extract frequently asked questions from document content.
    
    Parameters:
    - document_content: The content of the document
    - document_title: The title of the document
    
    Returns:
    - List of FAQ dictionaries with 'question' and 'answer' keys
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at extracting frequently asked questions from documents. Extract as many relevant questions and answers as possible from the provided content. IMPORTANT: Do not change, modify, or add any details that are not explicitly stated in the original document. Your answers must be factually identical to the information in the source material."},
                {"role": "user", "content": f"Document Title: {document_title}\n\nDocument Content:\n{document_content}\n\nExtract all possible FAQs from this document in JSON format with 'question' and 'answer' fields. Ensure your answers contain ONLY information that is explicitly stated in the document without any modifications or additions."}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        
        # Parse the response
        content = response.choices[0].message.content
        faqs_data = json.loads(content)
        
        # Ensure we have the expected format
        if "faqs" in faqs_data:
            return faqs_data["faqs"]
        else:
            # Handle case where the model didn't use the "faqs" key
            # Try to find any list of Q&A pairs
            for key, value in faqs_data.items():
                if isinstance(value, list) and len(value) > 0 and "question" in value[0]:
                    return value
            
            # Fallback
            print(f"Warning: Unexpected format in response for {document_title}")
            return []
            
    except Exception as e:
        print(f"Error processing document {document_title}: {str(e)}")
        return []

async def process_documents():
    """
    Processes all documents in the /documents folder and saves FAQs to /documents_parsed
    as text files
    """
    # Create output directory if it doesn't exist
    output_dir = Path("documents_parsed")
    output_dir.mkdir(exist_ok=True)
    
    # Get all files from documents directory
    documents_dir = Path("documents")
    if not documents_dir.exists():
        print("Error: /documents directory not found")
        return
    
    document_files = list(documents_dir.glob("**/*.*"))
    print(f"Found {len(document_files)} documents to process")
    
    for doc_path in document_files:
        # Skip hidden files and directories
        if doc_path.name.startswith('.'):
            continue
            
        print(f"Processing: {doc_path}")
        
        try:
            # Read document content
            with open(doc_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Extract FAQs
            faqs = await extract_faqs_from_document(content, doc_path.name)
            
            if faqs:
                # Save to output directory as txt file
                output_path = output_dir / f"{doc_path.stem}_faqs.txt"
                
                with open(output_path, 'w', encoding='utf-8') as out_file:
                    out_file.write(f"FAQs extracted from: {doc_path}\n")
                    out_file.write(f"Total FAQs: {len(faqs)}\n\n")
                    
                    for i, faq in enumerate(faqs, 1):
                        out_file.write(f"Q{i}: {faq['question']}\n")
                        out_file.write(f"A{i}: {faq['answer']}\n\n")
                
                print(f"✓ Saved {len(faqs)} FAQs to {output_path}")
            else:
                print(f"✗ No FAQs extracted from {doc_path}")
                
        except Exception as e:
            print(f"Error processing {doc_path}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(process_documents())


