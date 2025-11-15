# Ambedkar RAG System

A command-line Q&A system built for the Kalpit Pvt Ltd AI Intern Assignment. This system implements a Retrieval-Augmented Generation (RAG) pipeline to answer questions based on Dr. B.R. Ambedkar's speech using LangChain, ChromaDB, and Ollama.

## 🎯 Assignment Overview

This prototype demonstrates the fundamental building blocks of a RAG system:
- **Document Loading**: Ingests text from `speech.txt`
- **Text Chunking**: Splits text into manageable chunks
- **Embedding Generation**: Creates vector embeddings using HuggingFace models
- **Vector Storage**: Stores embeddings in ChromaDB (local vector store)
- **Retrieval**: Retrieves relevant chunks based on user questions
- **Generation**: Generates answers using Ollama with Mistral 7B

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **Framework**: LangChain
- **Vector Database**: ChromaDB (local, open-source)
- **Embeddings**: HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)
- **LLM**: Ollama with Mistral 7B
- **No API Keys Required**: 100% free and local

## 📋 Prerequisites

1. **Python 3.8+**
2. **Ollama**: Download and install Ollama
3. **Mistral Model**: Pull the Mistral 7B model

### Installing Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Mistral 7B model
ollama pull mistral
```

## 🚀 Quick Start

### 1. Clone/Download the Project

```bash
# If using git (replace with your actual repository)
git clone https://github.com/yourusername/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the System

#### Interactive Mode (Recommended)
```bash
python main.py
```

#### Single Question Mode
```bash
python main.py "What is the real remedy according to Ambedkar?"
```

## 💡 Usage Examples

### Interactive Mode
```
🤖 Ambedkar RAG System - Interactive Mode
Type your questions about Dr. Ambedkar's speech
Type 'quit', 'exit', or 'q' to exit
============================================================

❓ Ask a question: What does Ambedkar say is the real remedy?

🤔 Processing question: What does Ambedkar say is the real remedy?

📝 Answer:
--------------------------------------------------
The real remedy, according to Dr. Ambedkar, is to destroy the belief in the sanctity of the shastras. He argues that as long as people continue to hold the shastras as sacred and infallible, they will never be able to get rid of the caste system.
--------------------------------------------------

📚 Source Documents:

Source 1:
Content: The real remedy is to destroy the belief in the sanctity of the shastras. How do you expect to succeed if you allow the shastras to continue to be held as sacred and infallible?
```

### Sample Questions to Try
- "What is the real remedy for the caste problem?"
- "Why does Ambedkar compare social reform to gardening?"
- "What does Ambedkar say about the shastras?"
- "How does Ambedkar describe the problem of caste?"
- "What must people stop believing according to Ambedkar?"

## 📁 Project Structure

```
AmbedkarGPT-Intern-Task/
├── main.py              # Main RAG system implementation
├── requirements.txt     # Python dependencies
├── speech.txt          # Dr. Ambedkar's speech text
├── README.md           # This file
└── chroma_db/          # Local vector database (created automatically)
```

## 🔧 How It Works

### 1. Document Loading
- Uses `TextLoader` to load `speech.txt`
- Converts text into LangChain Document objects

### 2. Text Chunking
- `CharacterTextSplitter` splits text into 200-character chunks
- 50-character overlap ensures context continuity

### 3. Embedding Generation
- Uses `sentence-transformers/all-MiniLM-L6-v2` model
- Creates 384-dimensional vector embeddings
- Runs locally on CPU (no GPU required)

### 4. Vector Storage
- ChromaDB stores embeddings locally in `./chroma_db`
- Persistent storage - vectors are saved between sessions
- Automatic loading if database exists

### 5. Question Processing
- User question is embedded using the same model
- Similarity search finds top 3 most relevant chunks
- Retrieved context + question sent to LLM

### 6. Answer Generation
- Ollama with Mistral 7B processes the retrieved context
- Generates answer based solely on provided text
- Returns answer with source document references

## 🐛 Troubleshooting

### Common Issues

1. **Ollama not found**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull mistral
   ```

2. **Mistral model not pulled**
   ```bash
   ollama pull mistral
   ```

3. **Python version compatibility**
   - Ensure you're using Python 3.8 or higher
   - Check with: `python --version`

4. **Dependencies not installing**
   ```bash
   # Upgrade pip first
   pip install --upgrade pip
   # Then install requirements
   pip install -r requirements.txt
   ```

5. **Permission issues**
   ```bash
   # If you get permission errors, try:
   chmod +x main.py
   ```

### Performance Tips

- **First run**: May be slow as embeddings are generated and stored
- **Subsequent runs**: Much faster as vectors are loaded from cache
- **Memory usage**: System uses ~500MB RAM for embeddings and models

## 🧪 Testing the System

### Test Commands
```bash
# Test basic functionality
python main.py "What is the main problem discussed?"

# Test with specific quotes
python main.py "What does Ambedkar say about gardeners?"

# Test comprehension
python main.py "Why can't people have both caste practice and belief in shastras?"
```

### Expected Behavior
- System should answer questions based only on the provided text
- Answers should include relevant quotes and context
- Source documents should be displayed with each answer
- System should gracefully handle unrelated questions

## 📚 Key Concepts Demonstrated

1. **RAG Architecture**: Complete retrieval-augmented generation pipeline
2. **Vector Similarity**: Semantic search using embeddings
3. **Local Processing**: No external API dependencies
4. **Document Chunking**: Strategies for handling long texts
5. **Context Retrieval**: Finding relevant information for questions
6. **LLM Integration**: Connecting local LLM with retrieved context

## 🎓 Learning Outcomes

This assignment demonstrates understanding of:
- LangChain framework fundamentals
- Vector databases and embeddings
- RAG system architecture
- Local AI model deployment
- Command-line application development
- Error handling and user experience

## 📝 Notes for Reviewers

- **No API keys required**: Everything runs locally
- **Fully functional**: Complete end-to-end RAG pipeline
- **Well-commented**: Code includes comprehensive documentation
- **Error handling**: Graceful failure with helpful messages
- **User-friendly**: Interactive mode with clear instructions
- **Extensible**: Easy to add more documents or modify parameters

## 🤝 Contributing

This is an assignment submission, but the code is structured to be easily extensible for future development.

## 📄 License

This project is submitted as part of an assignment and is intended for educational purposes.

---

**Assignment Completed for: Kalpit Pvt Ltd AI Intern Hiring**  
**Candidate: [Your Name]**  
**Date: November 2025**
