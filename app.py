import gradio as gr
import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional
from vectorstore import get_rag_system

# Initialize RAG system
rag_system = get_rag_system()

# Store chat history per book
chat_histories = {}


def upload_book(file) -> Tuple[str, gr.update]:
    """Handle book upload."""
    if file is None:
        return "Please select a PDF file to upload.", gr.update()
    
    try:
        # Save uploaded file temporarily
        temp_path = file.name
        
        # Add book to RAG system
        book_id = rag_system.add_book(temp_path)
        book_name = rag_system.books[book_id]["name"]
        
        # Initialize chat history for this book
        chat_histories[book_id] = []
        
        # Update book list
        books = rag_system.get_books()
        book_choices = [f"{info['name']} ({book_id})" for book_id, info in books.items()]
        
        return f"✅ Book '{book_name}' uploaded successfully! You can now select it from the dropdown.", gr.update(
            choices=book_choices,
            value=book_choices[-1] if book_choices else None
        )
    except Exception as e:
        return f"❌ Error uploading book: {str(e)}", gr.update()


def get_book_list() -> gr.update:
    """Get updated list of books."""
    books = rag_system.get_books()
    if not books:
        return gr.update(choices=["No books available. Please upload a book first."], value=None)
    
    book_choices = [f"{info['name']} ({book_id})" for book_id, info in books.items()]
    return gr.update(choices=book_choices, value=book_choices[0] if book_choices else None)


def extract_book_id(book_selection: str) -> Optional[str]:
    """Extract book ID from selection string."""
    if not book_selection or "No books available" in book_selection:
        return None
    # Format: "Book Name (book_id)"
    if "(" in book_selection and ")" in book_selection:
        return book_selection.split("(")[-1].rstrip(")")
    return None


def chat_with_book(message: str, history: list, book_selection: str) -> list:
    """Handle chat with selected book."""
    book_id = extract_book_id(book_selection)
    
    if not book_id:
        history.append((message, "Please select a book first or upload a new book."))
        return history
    
    if book_id not in chat_histories:
        chat_histories[book_id] = []
    
    # Add user message to history
    chat_histories[book_id].append((message, None))
    
    # Get response from RAG system
    try:
        response = rag_system.query_book(book_id, message)
        chat_histories[book_id][-1] = (message, response)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        chat_histories[book_id][-1] = (message, error_msg)
    
    # Return updated history
    return chat_histories[book_id].copy()


def load_chat_history(book_selection: str) -> list:
    """Load chat history for selected book."""
    book_id = extract_book_id(book_selection)
    if book_id and book_id in chat_histories:
        return chat_histories[book_id]
    return []


def delete_book(book_selection: str) -> Tuple[str, gr.update, list]:
    """Delete selected book."""
    book_id = extract_book_id(book_selection)
    
    if not book_id:
        return "Please select a book to delete.", gr.update(), []
    
    try:
        book_name = rag_system.books[book_id]["name"]
        success = rag_system.delete_book(book_id)
        
        if success:
            # Remove chat history
            if book_id in chat_histories:
                del chat_histories[book_id]
            
            # Update book list
            books = rag_system.get_books()
            if books:
                book_choices = [f"{info['name']} ({bid})" for bid, info in books.items()]
                selected = book_choices[0] if book_choices else None
            else:
                book_choices = ["No books available. Please upload a book first."]
                selected = None
            
            return f"✅ Book '{book_name}' deleted successfully.", gr.update(
                choices=book_choices,
                value=selected
            ), []
        else:
            return f"❌ Failed to delete book '{book_name}'.", gr.update(), []
    except Exception as e:
        return f"❌ Error deleting book: {str(e)}", gr.update(), []


# Create Gradio interface
with gr.Blocks(title="Book RAG Chat System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 Book RAG Chat System
        
        Upload PDF books and chat with them using AI-powered RAG (Retrieval-Augmented Generation).
        Select a book from the dropdown to start chatting!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Book")
            file_upload = gr.File(
                label="Upload PDF Book",
                file_types=[".pdf"],
                type="filepath"
            )
            upload_btn = gr.Button("Upload Book", variant="primary")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
            
            gr.Markdown("### 📖 Select Book")
            book_dropdown = gr.Dropdown(
                label="Available Books",
                choices=["No books available. Please upload a book first."],
                value=None,
                interactive=True
            )
            refresh_books_btn = gr.Button("Refresh Book List", size="sm")
            delete_book_btn = gr.Button("Delete Selected Book", variant="stop", size="sm")
            delete_status = gr.Textbox(label="Delete Status", interactive=False)
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with Book")
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_copy_button=True
            )
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about the selected book...",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")
    
    # Event handlers
    upload_btn.click(
        fn=upload_book,
        inputs=[file_upload],
        outputs=[upload_status, book_dropdown]
    )
    
    refresh_books_btn.click(
        fn=get_book_list,
        outputs=[book_dropdown]
    )
    
    book_dropdown.change(
        fn=load_chat_history,
        inputs=[book_dropdown],
        outputs=[chatbot]
    )
    
    delete_book_btn.click(
        fn=delete_book,
        inputs=[book_dropdown],
        outputs=[delete_status, book_dropdown, chatbot]
    )
    
    def user_message(message, history, book_selection):
        return "", chat_with_book(message, history, book_selection)
    
    msg.submit(
        fn=user_message,
        inputs=[msg, chatbot, book_dropdown],
        outputs=[msg, chatbot]
    )
    
    submit_btn.click(
        fn=user_message,
        inputs=[msg, chatbot, book_dropdown],
        outputs=[msg, chatbot]
    )
    
    def clear_chat(book_selection):
        """Clear chat history for selected book."""
        book_id = extract_book_id(book_selection)
        if book_id and book_id in chat_histories:
            chat_histories[book_id] = []
        return []
    
    clear_btn.click(
        fn=clear_chat,
        inputs=[book_dropdown],
        outputs=[chatbot]
    )
    
    # Load books on startup
    demo.load(
        fn=get_book_list,
        outputs=[book_dropdown]
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
