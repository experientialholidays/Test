import gradio as gr
import asyncio
import os
from dotenv import load_dotenv
from agents import Runner, trace, gen_trace_id
from vector_db import VectorDBManager
from db import SessionDBManager
from session_handler import SessionHandler

# --- KEY CHANGE: Import the CORE functions ---
from auroville_agent import (
    auroville_agent, 
    db_manager, 
    initialize_retriever, 
    get_event_details_core,  # The plain python function
    get_daily_events_core,   # The plain python function
    EVENT_DATA_STORE
)

import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

SESSION_DB_FILE = "sessions.db"
session_db_manager = SessionDBManager(db_file=SESSION_DB_FILE)
session_handler = SessionHandler(session_db_manager=session_db_manager)

# --- VECTOR DB INIT ---
VECTOR_DB_NAME = "vector_db"
DB_FOLDER = "input"
try:
    print("--- STARTING VECTOR DB INITIALIZATION ---")
    vectorstore = db_manager.create_or_load_db(force_refresh=False)
    initialize_retriever(vectorstore)
    print("--- VECTOR DB INITIALIZATION COMPLETE ---")
except Exception as e:
    logger.error(f"FATAL ERROR during DB initialization: {e}")
    pass

# Helpers
DETAILS_RE = re.compile(r'^\s*details\(\s*(\d+)\s*\)\s*$', re.IGNORECASE)
DIGITS_RE = re.compile(r'^\s*(\d+)\s*$')
SHOW_DAILY_ALIASES = {
    "show daily events", "show daily event", "show daily",
    "show daily events please", "show daily events, please",
    "show daily events yes", "show daily events y",
}

# ----------------------------------------------------------
# HISTORY CONVERTER HELPER FUNCTION
# ----------------------------------------------------------
def history_to_agent_format(history, new_q):
    """Converts Gradio's history (list of lists/tuples) to the Agent's message format (list of dicts)."""
    agent_messages = []
    
    # history from Gradio is a list of [user_msg, assistant_msg] pairs
    for user_msg, assistant_msg in history:
        # Filter out messages from the Gradio history that might be None or empty strings
        if user_msg:
            agent_messages.append({"role": "user", "content": str(user_msg)})
        if assistant_msg:
            agent_messages.append({"role": "assistant", "content": str(assistant_msg)})

    # Add the current user question
    if new_q:
        agent_messages.append({"role": "user", "content": str(new_q)})
        
    return agent_messages

# ----------------------------------------------------------
# SESSION INITIALIZATION FUNCTION (FOR PERSISTENCE)
# ----------------------------------------------------------
def initialize_session_on_load():
    """Retrieves the last active session ID and history from the database."""
    # NOTE: You MUST ensure your session_handler has a method like get_last_session_data
    # This function provides the necessary outputs to populate the Gradio components on load.
    
    # Placeholder implementation (Replace with your actual SessionHandler logic)
    last_session_id, history_list = session_handler.get_last_session_data() 
    
    # Outputs: session_id_state, session_id_bridge, chatbot
    return last_session_id, last_session_id, history_list


# ----------------------------------------------------------
# CHAT FUNCTION
# ----------------------------------------------------------
async def streaming_chat(question, history, session_id):
    
    # 1. Sanitization 
    if isinstance(question, dict):
        q_raw = question.get('text', "")
    else:
        q_raw = question or ""
    q = q_raw.strip()

    logger.info(f'Processing chat for session: {session_id} | question: {q!r}')
    if not session_id or session_id == "null": return

    # 2. Routing (Uses _core functions, bypasses Agent Tools)
    
    # Check details(...)
    m = DETAILS_RE.match(q)
    if m:
        idx = int(m.group(1))
        result = get_event_details_core(f"details({idx})")
        
        session_handler.save_message(session_id, "user", q)
        session_handler.save_message(session_id, "assistant", result)
        updated_history = history.copy()
        updated_history.append([q, result]) 
        yield updated_history
        return

    # Check Plain Integer
    m2 = DIGITS_RE.match(q)
    if m2:
        idx = int(m2.group(1))
        result = get_event_details_core(str(idx))
        
        session_handler.save_message(session_id, "user", q)
        session_handler.save_message(session_id, "assistant", result)
        updated_history = history.copy()
        updated_history.append([q, result])
        yield updated_history
        return

    # Check Daily Events
    if q.lower() in SHOW_DAILY_ALIASES:
        try: last_index = max(EVENT_DATA_STORE.keys()) if EVENT_DATA_STORE else 0
        except: last_index = 0
        
        result = get_daily_events_core(start_number=last_index)
        
        session_handler.save_message(session_id, "user", q)
        session_handler.save_message(session_id, "assistant", result)
        updated_history = history.copy()
        updated_history.append([q, result])
        yield updated_history
        return

    # 3. LLM Flow (Updated with history conversion and robust stream processing)
    session_handler.save_message(session_id, "user", q)

    try:
        response_text = ""
        
        # --- CRITICAL FIX: CONVERT GRADIO HISTORY TO AGENT FORMAT ---
        clean_message = history_to_agent_format(history, q)
        # --- END CRITICAL FIX ---
        
        trace_id = gen_trace_id()
        with trace("Auroville chatbot", trace_id=trace_id):
            result = Runner.run_streamed(auroville_agent, clean_message)
            
            # Use the existing history as the base for the Gradio output
            updated_history = history.copy()
            updated_history.append([q, response_text]) # Pre-add the new entry

            # Robust stream processing logic (as previously corrected)
            async for event in result.stream_events():
                
                delta = None
                
                # Case 1: The event object itself has the text delta (safe check with getattr)
                if hasattr(event, 'response_text_delta'):
                    delta = getattr(event, 'response_text_delta', None)
                
                # Case 2: The event follows the original nested structure
                elif type(event).__name__ == "RawResponsesStreamEvent":
                    data = getattr(event, 'data', None)
                    if data and data.__class__.__name__ == "ResponseTextDeltaEvent":
                        delta = getattr(data, 'delta', None)
                        
                # Check if we successfully extracted a text delta
                if delta:
                    response_text += delta
                    
                    # Update the last entry in the Gradio history list
                    updated_history[-1][1] = response_text
                    yield updated_history

        if response_text:
            session_handler.save_message(session_id, "assistant", response_text)
        else:
            error_msg = "I couldn't generate a response."
            updated_history[-1][1] = error_msg
            yield updated_history
            session_handler.save_message(session_id, "assistant", error_msg)

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        updated_history = history.copy()
        updated_history.append([q, error_msg])
        yield updated_history
        session_handler.save_message(session_id, "assistant", error_msg)


# JS Code
JS_CODE = """
function attachClickHandlers(msg_input_id, submit_btn_id) {
    document.addEventListener('click', function(event) {
        const anchor = event.target.closest &&
            event.target.closest('a[href^="#DETAILS::"], a[href^="#SHOWDAILY::"]');
        if (!anchor) return;

        event.preventDefault();
        event.stopImmediatePropagation();
        event.stopPropagation();

        const href = anchor.getAttribute('href');
        const submitBtn = document.getElementById(submit_btn_id);
        const inputContainer = document.getElementById(msg_input_id);
        const msgInput = inputContainer ? inputContainer.querySelector('textarea') : null;

        if (!msgInput || !submitBtn) return;

        let textToSend = "";
        if (href.startsWith("#DETAILS::")) {
            const parts = href.substring(1).split("::");
            const match = parts[1].match(/(\\d+)/);
            if (match) textToSend = "details(" + match[1] + ")";
        } else if (href.startsWith("#SHOWDAILY::")) {
            const parts = href.substring(1).split("::");
            if (parts[1] === "YES") textToSend = "show daily events";
            else if (parts[1] === "NO") textToSend = "no";
        }

        if (textToSend) {
            msgInput.value = textToSend;
            msgInput.dispatchEvent(new Event('input', { bubbles: true }));
            setTimeout(() => { submitBtn.click(); }, 200);
        }
    });
}
"""

if __name__ == "__main__":
    with gr.Blocks() as demo:
         gr.Markdown("# 🤖 Auroville Events Chatbot")
         session_id_state = gr.State("")
         session_id_bridge = gr.Textbox(value="", visible=False)
         temp_storage_state = gr.State("")
         chatbot = gr.Chatbot(height=500, value=[])
         
         msg = gr.Textbox(placeholder="Ask me anything...", lines=1, label="Message", show_label=False, elem_id="msg_input_field")
         with gr.Row():
            submit = gr.Button("Send", variant="primary", elem_id="submit_button")
            new_session_btn = gr.Button("New Session")

         # --- MODIFIED LOAD LINE FOR PERSISTENCE ---
         demo.load(
             initialize_session_on_load, 
             None, 
             [session_id_state, session_id_bridge, chatbot], 
             js=f"() => {{ {JS_CODE} attachClickHandlers('msg_input_field', 'submit_button'); }}"
         )
         # --- END MODIFIED LOAD LINE ---
         
         session_handler.setup_session_handlers(demo, session_id_state, session_id_bridge, temp_storage_state, chatbot, new_session_btn)

         msg.submit(streaming_chat, inputs=[msg, chatbot, session_id_state], outputs=[chatbot]).then(lambda: "", None, msg)
         submit.click(streaming_chat, inputs=[msg, chatbot, session_id_state], outputs=[chatbot]).then(lambda: "", None, msg)

    server_port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=server_port, inbrowser=False, debug=False)
