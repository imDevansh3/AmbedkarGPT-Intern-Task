#!/usr/bin/env python3
"""
Ambedkar RAG System -  Q&A System
Assignment 1 for Kalpit Pvt Ltd AI Intern Hiring

This system implements a Retrieval-Augmented Generation (RAG) pipeline using:
- LangChain for orchestration
- ChromaDB for vector storage
- HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)
- Ollama with Mistral 7B for LLM inference
"""

import os
import sys
from typing import List

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document


# Global variables
#


def setup_embeddings():
    """Setup HuggingFace embeddings model"""
    global embeddings_model
    print("Setting up HuggingFace embeddings...")
    try:
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("Embeddings model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return False


def load_and_process_documents(speech_file="speech.txt", persist_directory="./chroma_db"):
    """Load text, split into chunks, and create vector store"""
    global vector_store
    print(f"Loading document from {speech_file}...")
    
    # Check if file exists
    if not os.path.exists(speech_file):
        print(f"Error: {speech_file} not found!")
        return False
    
    try:
        # Load text document
        loader = TextLoader(speech_file)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s)")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separator=" "
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split text into {len(chunks)} chunks")
        
        # Create or load vector store
        if os.path.exists(persist_directory):
            print("Loading existing vector store...")
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings_model
            )
        else:
            print("Creating new vector store...")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings_model,
                persist_directory=persist_directory
            )
            vector_store.persist()
        print("Vector store ready")
        return True
        
    except Exception as e:
        print(f"Error processing documents: {e}")
        return False


def setup_llm_and_chain():
    """Setup Ollama LLM and QA chain"""
    global qa_chain
    print("Setting up Ollama with Mistral 7B...")
    try:
        # Initialize Ollama LLM
        llm = Ollama(model="mistral")
        print("Ollama Mistral 7B loaded")
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        print("QA chain initialized")
        return True
        
    except Exception as e:
        print(f"Error setting up LLM: {e}")
        return False


def query_system(question):
    """Query the RAG system with a question"""
    try:
        print(f"\nProcessing question: {question}")
        result = qa_chain({"query": question})
        
        print("\nAnswer:")
        print("-" * 50)
        print(result["result"])
        print("-" * 50)
        
        print("\nSource Documents:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"\nSource {i}:")
            print(f"Content: {doc.page_content.strip()}")
        
        return result
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return {"error": str(e)}


def interactive_mode():
    """Run the system in interactive command-line mode"""
    print("\n" + "="*60)
    print("Ambedkar RAG System - Interactive Mode")
    print("Type your questions about Dr. Ambedkar's speech")
    print("Type 'quit', 'exit', or 'q' to exit")
    print("="*60)
    
    while True:
        try:
            question = input("\nAsk a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                print("Please enter a valid question.")
                continue
            
            query_system(question)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def initialize_system():
    """Initialize all system components"""
    print("Starting Ambedkar RAG System...")
    print("Note: Make sure Ollama is installed and Mistral model is available")
    
    # Setup components step by step
    if not setup_embeddings():
        print("Failed to setup embeddings. Please check your installation.")
        return False
    
    if not load_and_process_documents():
        print("Failed to load documents. Please check speech.txt file.")
        return False
    
    if not setup_llm_and_chain():
        print("Failed to setup LLM. Please check Ollama installation.")
        return False
    
    print("System initialization complete!")
    return True


def main():
    """Main function to run the RAG system"""
    if initialize_system():
        interactive_mode()
    else:
        print("System failed to initialize. Please check the error messages above.")


if __name__ == "__main__":
    main()
