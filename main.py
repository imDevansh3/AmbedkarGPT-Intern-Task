#!/usr/bin/env python3
"""
Ambedkar RAG System - Command Line Q&A System
Assignment 1 for Kalpit Pvt Ltd AI Intern Hiring

This system implements a simple RAG pipeline using:
- LangChain
- ChromaDB
- HuggingFace embeddings (MiniLM-L6-v2)
- Ollama (Mistral 7B)
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document





def setup_embeddings():
    """Initialize HuggingFace embedding model."""
    print("Setting up HuggingFace embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("Embeddings loaded successfully.")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None


def load_and_process_documents(embeddings, speech_file="speech.txt", persist_directory="./chroma_db"):
    """Load, split, and embed documents in ChromaDB."""
    print(f"Loading document from {speech_file}...")

    if not os.path.exists(speech_file):
        print("Error: speech.txt not found.")
        return None

    try:
        loader = TextLoader(speech_file)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s).")

        splitter = CharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separator=" "
        )
        chunks = splitter.split_documents(documents)
        print(f"Generated {len(chunks)} chunks.")

        if os.path.exists(persist_directory):
            print("Loading existing ChromaDB vector store...")
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        else:
            print("Creating new ChromaDB vector store...")
            vector_store = Chroma.from_documents(
                chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            vector_store.persist()

        print("Vector store ready.")
        return vector_store

    except Exception as e:
        print(f"Error processing documents: {e}")
        return None


def setup_llm_and_chain(vector_store):
    """Create Ollama LLM and attach it to a RetrievalQA chain."""
    print("Initializing Ollama Mistral 7B...")
    try:
        llm = Ollama(model="mistral")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        print("QA chain created successfully.")
        return qa_chain

    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None


def query_system(qa_chain, question):
    """Ask a question to the RAG system."""
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
            print(doc.page_content.strip())

    except Exception as e:
        print(f"Error answering question: {e}")


def interactive_mode(qa_chain):
    """CLI loop for continuous Q&A."""
    print("\n" + "="*60)
    print("Ambedkar RAG System - Interactive Mode")
    print("Ask questions based only on the speech.")
    print("Type 'quit', 'exit', or 'q' to exit.")
    print("="*60)

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if question:
            query_system(qa_chain, question)
        else:
            print("Please enter a valid question.")


def main():
    """Main runner: initializes components and starts Q&A."""
    print("Starting Ambedkar RAG System...\n")

    embeddings = setup_embeddings()
    if embeddings is None:
        return

    vector_store = load_and_process_documents(embeddings)
    if vector_store is None:
        return

    qa_chain = setup_llm_and_chain(vector_store)
    if qa_chain is None:
        return

    interactive_mode(qa_chain)


if __name__ == "__main__":
    main()