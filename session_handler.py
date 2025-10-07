import uuid
import gradio as gr


class SessionHandler:
    """
    Handles session management for Gradio UI including:
    - LocalStorage-based session persistence (works with share=True)
    - Chat history loading and formatting
    - Session creation and restoration
    """
    
    def __init__(self, session_db_manager):
        """
        Initialize the SessionHandler with a database manager.
        
        Args:
            session_db_manager: Instance of SessionDBManager from db.py
        """
        self.session_db = session_db_manager
    
    def load_chat_history(self, session_id):
        """
        Load chat history from database in Gradio messages format.
        (DB already returns [{"role": ..., "content": ...}])
        """
        print(f'Loading chat history for session: {session_id}')
        history = self.session_db.load_history(session_id)
        print(f'Loaded {len(history)} messages from database')
        return history
        
    def save_message(self, session_id, role, content):
        """
        Save a message to the database.
        
        Args:
            session_id: Unique session identifier
            role: Either "user" or "assistant"
            content: The message content
        """
        self.session_db.save_message(session_id, role, content)
    
    def format_history_for_prompt(self, history):
        """
        Convert Gradio chatbot history (list of dicts) into a string
        suitable for LLM prompt.
        """
        formatted = []
        for msg in history:
            if msg["role"] == "user":
                formatted.append(f"Human: {msg['content']}")
            elif msg["role"] == "assistant":
                formatted.append(f"Assistant: {msg['content']}")
        return "\n".join(formatted)
    
    def get_or_create_session(self, stored_session_id: str):
        """
        Get session ID from localStorage (via JS) or create a new one.
        Also loads chat history for existing sessions.
        
        Args:
            stored_session_id: Session ID from localStorage (passed from JS)
            
        Returns:
            Tuple of (session_id, chat_history)
        """
        print("=== get_or_create_session called ===")
        print(f"Received session_id from localStorage: '{stored_session_id}'")
        
        if not stored_session_id or stored_session_id == "undefined" or stored_session_id == "null" or stored_session_id.strip() == "":
            # Create new session ID
            session_id = str(uuid.uuid4())
            print(f"New session created: {session_id}")
            return session_id, []
        else:
            print(f"Existing session loaded: {stored_session_id}")
            # Load history from database
            history = self.load_chat_history(stored_session_id)
            return stored_session_id, history
    
    def create_new_session(self):
        """
        Create a new session and clear the chat.
        
        Returns:
            Tuple of (new_session_id, empty_chat_history)
        """
        session_id = str(uuid.uuid4())
        print(f"Manual new session created: {session_id}")
        return session_id, []
    
    @staticmethod
    def get_localStorage_reader_js():
        """
        Returns JavaScript code to read session_id from localStorage.
        
        Returns:
            JavaScript code as string
        """
        return """
        () => {
            const stored = localStorage.getItem("session_id");
            console.log("=== Reading from localStorage ===");
            console.log("Stored session_id:", stored);
            return stored || "";
        }
        """
    
    @staticmethod
    def get_localStorage_setter_js():
        """
        Returns JavaScript code to set session_id in localStorage.
        
        Returns:
            JavaScript code as string
        """
        return """
        (session_id) => {
            console.log("=== Setting localStorage ===");
            console.log("Session ID received:", session_id);
            console.log("Session ID type:", typeof session_id);
            if (session_id && session_id !== 'null' && session_id !== 'undefined' && session_id.trim() !== '') {
                localStorage.setItem("session_id", session_id);
                console.log("✓ localStorage set successfully");
                console.log("Current localStorage session_id:", localStorage.getItem("session_id"));
            } else {
                console.error("✗ Invalid session_id:", session_id);
            }
        }
        """
    
    @staticmethod
    def get_new_session_localStorage_js():
        """
        Returns JavaScript code to set localStorage for new session.
        
        Returns:
            JavaScript code as string
        """
        return """
        (session_id) => {
            console.log("=== New Session - Setting localStorage ===");
            if (session_id && session_id !== 'null' && session_id !== 'undefined') {
                localStorage.setItem("session_id", session_id);
                console.log("✓ New session localStorage set:", session_id);
            }
        }
        """
    
    def setup_session_handlers(self, demo, session_id_state, session_id_bridge, temp_storage_state, chatbot, new_session_btn):
        """
        Set up all session-related event handlers for the Gradio app.
        Uses localStorage instead of cookies (works with share=True).
        
        Args:
            demo: Gradio Blocks app
            session_id_state: gr.State component for session ID
            session_id_bridge: gr.Textbox component (hidden) for passing to JS
            temp_storage_state: gr.State component for receiving localStorage value (not used in new approach)
            chatbot: gr.Chatbot component
            new_session_btn: gr.Button component for creating new session
        """
        # DIRECT APPROACH: Initialize session on page load
        # This uses a dummy textbox to trigger the chain reliably
        demo.load(
            fn=self._initialize_session_with_dummy,
            inputs=[session_id_bridge],
            outputs=[session_id_state, chatbot, session_id_bridge],
            js="""
            () => {
                const stored = localStorage.getItem("session_id");
                console.log("=== Page Load: Reading localStorage ===");
                console.log("Stored session_id:", stored);
                return stored || "";
            }
            """
        )
        
        # Save to localStorage after initialization
        session_id_bridge.change(
            fn=lambda x: x,   # dummy passthrough
            inputs=[session_id_bridge],
            outputs=[session_id_bridge],
            js=self.get_localStorage_setter_js()
        )
        
        # Create new session button handler
        new_session_btn.click(
            fn=self.create_new_session,
            inputs=[],
            outputs=[session_id_state, chatbot]
        ).then(
            fn=lambda x: (x,), 
            inputs=[session_id_state],
            outputs=[session_id_bridge]
        )
    
    def _initialize_session_with_dummy(self, stored_session_id: str = ""):
        """
        Internal method to initialize session. 
        This is called by demo.load() with the stored session ID from JS.
        
        Args:
            stored_session_id: Session ID from localStorage (passed from JS return value)
            
        Returns:
            Tuple of (session_id, chat_history, session_id_for_bridge)
        """
        print("=== _initialize_session_with_dummy called ===")
        print(f"Received from JS: '{stored_session_id}'")
        
        # Call the existing get_or_create_session logic
        session_id, chat_history = self.get_or_create_session(stored_session_id)
        
        # Return session_id twice: once for state, once for bridge
        return session_id, chat_history, session_id