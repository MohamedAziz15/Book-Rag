# 📚 Book RAG Chat System

A powerful Retrieval-Augmented Generation (RAG) system that allows you to upload PDF books and chat with them using AI. Each book is stored separately, and you can select which book to chat with.

## Features

- 📤 **Upload PDF Books**: Upload multiple PDF books to the system
- 📖 **Book Selection**: Select which book you want to chat with from a dropdown
- 💬 **AI-Powered Chat**: Ask questions about the book content using RAG
- 🗑️ **Book Management**: Delete books you no longer need
- 💾 **Persistent Storage**: Books are stored in a vector database for fast retrieval

## Requirements

- Python 3.12+
- Google Gemini API key (for embeddings and LLM)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Book-Rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

### Running the Application

Start the Gradio web interface:
```bash
python app.py
```

The application will start on `http://localhost:7860` (or the URL shown in the terminal).

### Using the Web Interface

1. **Upload a Book**:
   - Click "Choose File" under "Upload Book"
   - Select a PDF file
   - Click "Upload Book"
   - Wait for the upload confirmation

2. **Select a Book**:
   - Use the dropdown menu under "Select Book"
   - Choose the book you want to chat with

3. **Chat with the Book**:
   - Type your question in the text box
   - Click "Send" or press Enter
   - The AI will answer based on the book's content

4. **Manage Books**:
   - Click "Refresh Book List" to update the list
   - Click "Delete Selected Book" to remove a book
   - Click "Clear Chat" to clear the current conversation

## Architecture

### Components

- **`vectorstore.py`**: Contains the `BookRAGSystem` class that manages:
  - Multiple book vector stores (one per book)
  - PDF loading and chunking
  - RAG query processing
  - Book metadata management

- **`app.py`**: Gradio web interface with:
  - File upload functionality
  - Book selection dropdown
  - Chat interface
  - Book management features

- **`gemini_llm.py`**: Google Gemini LLM configuration

### How It Works

1. **Upload**: When you upload a PDF, it's:
   - Loaded and split into chunks
   - Embedded using Google Gemini embeddings
   - Stored in a separate Chroma collection per book

2. **Query**: When you ask a question:
   - The system retrieves relevant chunks from the selected book
   - The LLM generates an answer based on the retrieved context
   - The answer is displayed in the chat interface

## Technical Details

- **Vector Store**: ChromaDB with persistent storage
- **Embeddings**: Google Gemini Embedding Model (`models/gemini-embedding-001`)
- **LLM**: Google Gemini (`gemini-2.5-flash-lite`)
- **Chunking**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Retrieval**: Top-k similarity search (k=4 by default)

## File Structure

```
Book-Rag/
├── app.py                 # Gradio web interface
├── vectorstore.py         # RAG system core logic
├── gemini_llm.py          # LLM configuration
├── main.py                # Legacy script (can be removed)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── chroma_langchain_db/   # Vector database storage
└── docs/                  # Sample PDF files
```

## Troubleshooting

### "Book not found" error
- Make sure you've selected a book from the dropdown
- Try refreshing the book list

### Upload fails
- Ensure the file is a valid PDF
- Check that you have write permissions in the directory
- Verify your Google API key is set correctly

### No response from AI
- Check your internet connection
- Verify your Google API key is valid and has quota
- Check the console for error messages

## License

This project is open source and available for use.

## Contributing

Feel free to submit issues and enhancement requests!