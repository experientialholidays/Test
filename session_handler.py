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
        
        NOTE: This function needs to convert the DB format ([{"role":..., "content":...}])
        to the Gradio chatbot format (list of [user_msg, assistant_msg] pairs).
        
        Since the DB messages are saved one-by-one, they need to be paired up here.
        Assuming the DB returns a list of dictionaries like:
        [{"role": "user", "content": "..."}]
        
        To work with the rest of your app, the output must be:
        [[user_msg_1, assistant_msg_1], [user_msg_2, assistant_msg_2], ...]
        """
        print(f'Loading chat history for session: {session_id}')
        db_history = self.session_db.load_history(session_id)
        print(f'Loaded {len(db_history)} messages from database')
        
        # Convert DB message list to Gradio Chatbot pair list
        gradio_history = []
        current_pair = ["", ""] # [User, Assistant]
        
        for msg in db_history:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "user":
                # Start a new pair
                if current_pair[0] != "" and current_pair[1] != "":
                    gradio_history.append(current_pair)
                    current_pair = ["", ""]
                current_pair[0] = content
            elif role == "assistant":
                current_pair[1] = content
                
        # Append the final pair if non-empty
        if current_pair[0] != "" or current_pair[1] != "":
            gradio_history.append(current_pair)
            
        return gradio_history
        
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
        
        NOTE: Based on your app.py fix, this function is likely unused now,
        but keeping it for completeness. The `history_to_agent_format` helper
        in app.py handles this better.
        """
        formatted = []
        # Assuming history here is the list of dicts from the old format, 
        # but your app.py now passes list of lists.
        for msg in history:
            # Check if history is the Gradio list of pairs or the old dicts list
            if isinstance(msg, dict) and msg.get("role") == "user":
                formatted.append(f"Human: {msg['content']}")
            elif isinstance(msg, dict) and msg.get("role") == "assistant":
                formatted.append(f"Assistant: {msg['content']}")
            # If it's Gradio's [user, assistant] list of lists:
            elif isinstance(msg, list) and len(msg) == 2:
                if msg[0]: formatted.append(f"Human: {msg[0]}")
                if msg[1]: formatted.append(f"Assistant: {msg[1]}")
                
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
        
        if not stored_session_id or stored_session_id in ["undefined", "null"] or stored_session_id.strip() == "":
            # Create new session ID
            session_id = str(uuid.uuid4())
            print(f"New session created: {session_id}")
            return session_id, []
        else:
            print(f"Existing session loaded: {stored_session_id}")
            # Load history from database (history is in Gradio format)
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
            if (session_id && session_id !== 'null' && session_id !== 'undefined' && session_id.trim() !== '') {
                localStorage.setItem("session_id", session_id);
                console.log("✓ localStorage set successfully");
            } else {
                console.error("✗ Invalid session_id:", session_id);
            }
        }
        """
    
    def setup_session_handlers(self, demo, session_id_state, session_id_bridge, temp_storage_state, chatbot, new_session_btn):
        """
        Set up all session-related event handlers for the Gradio app.
        
        Args:
            demo: Gradio Blocks app
            session_id_state: gr.State component for session ID
            session_id_bridge: gr.Textbox component (hidden) for passing to JS
            temp_storage_state: gr.State component (unused in this approach)
            chatbot: gr.Chatbot component
            new_session_btn: gr.Button component for creating new session
        """
        # --- 1. INITIAL SESSION LOADING (ON PAGE LOAD) ---
        # 1a. On page load, run JS to fetch session_id from localStorage and pass it to Python.
        # The output of the JS runs the Python function _initialize_session_with_dummy.
        demo.load(
            # fn: The Python function to call with the JS output
            fn=self._initialize_session_with_dummy,
            # inputs: The component that receives the JS output (session_id_bridge is the input receiver)
            inputs=[session_id_bridge],
            # outputs: The components to update after Python runs
            outputs=[session_id_state, chatbot, session_id_bridge],
            # js: The JavaScript code that runs first and returns the session_id
            js=self.get_localStorage_reader_js()
        )
        
        # --- 2. LOCALSTORAGE SETTER (AFTER ANY SESSION ID CHANGE) ---
        # Whenever session_id_bridge changes (after load or new session click), update localStorage.
        session_id_bridge.change(
            fn=lambda x: x,   # Python passthrough (returns its input)
            inputs=[session_id_bridge],
            outputs=[session_id_bridge],
            js=self.get_localStorage_setter_js()
        )
        
        # --- 3. CREATE NEW SESSION HANDLER ---
        new_session_btn.click(
            fn=self.create_new_session,
            inputs=[],
            outputs=[session_id_state, chatbot]
        ).then(
            # Pass the new session_id (from session_id_state) to the bridge to trigger localStorage update
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
            Tuple of (session_id_state, chat_history, session_id_bridge)
        """
        print("=== _initialize_session_with_dummy called ===")
        print(f"Received from JS: '{stored_session_id}'")
        
        # Call the existing get_or_create_session logic
        session_id, chat_history = self.get_or_create_session(stored_session_id)
        
        # Return session_id twice: once for state, once for bridge (to trigger localStorage update)
        return session_id, chat_history, session_id
