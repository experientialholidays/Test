import gradio as gr
import asyncio
import os
import urllib.parse
from dotenv import load_dotenv
from agents import Runner, trace, gen_trace_id
from db import SessionDBManager
from session_handler import SessionHandler
import logging

# --- IMPORTS FROM AGENT FILE ---
# We import the Cache (EVENT_DATA_STORE) and the Formatter
from auroville_agent import (
    auroville_agent, 
    EVENT_DATA_STORE, 
    format_event_card, 
    db_manager
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)
SESSION_DB_FILE = "sessions.db"
session_db_manager = SessionDBManager(db_file=SESSION_DB_FILE)
session_handler = SessionHandler(session_db_manager=session_db_manager)

# ---------------------------------------------------------
#  JAVASCRIPT: Intercepts click, sends command
# ---------------------------------------------------------
custom_js = """
function handleEventLinks() {
    document.addEventListener('click', function(e) {
        const target = e.target;
        const link = target.closest('a');
        
        if (link && link.getAttribute('href') && link.getAttribute('href').startsWith('#FETCH::')) {
            e.preventDefault(); 
            const href = link.getAttribute('href');
            const encodedTitle = href.split('::')[1];
            
            // Send Command
            const command = "__FETCH__::" + encodedTitle;

            const textarea = document.querySelector('#chat-input textarea');
            const submitBtn = document.querySelector('#submit-btn');

            if (textarea && submitBtn) {
                textarea.value = command;
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                setTimeout(() => { submitBtn.click(); }, 100);
            }
        }
    });
}
"""

# ---------------------------------------------------------
#  CACHED FETCH LOGIC
# ---------------------------------------------------------
def get_cached_details(encoded_title):
    """
    1. Checks RAM Cache (EVENT_DATA_STORE).
    2. If missing (restart/expire), fallback to DB search.
    """
    try:
        # 1. RAM CACHE CHECK
        if encoded_title in EVENT_DATA_STORE:
            logger.info(f"⚡ CACHE HIT: Found '{encoded_title}' in memory.")
            doc = EVENT_DATA_STORE[encoded_title]
            return format_event_card(doc.metadata, doc.page_content)

        # 2. FALLBACK (DB Search)
        logger.info(f"⚠️ CACHE MISS: Fetching '{encoded_title}' from DB.")
        title = urllib.parse.unquote(encoded_title)
        docs = db_manager.vectorstore.similarity_search(title, k=1)
        
        if docs:
            # Update cache for next time
            EVENT_DATA_STORE[encoded_title] = docs[0]
            return format_event_card(docs[0].metadata, docs[0].page_content)
        
        return "Sorry, I couldn't retrieve the details for that event."
        
    except Exception as e:
        return f"Error retrieving details: {str(e)}"


# ---------------------------------------------------------
#  CHAT LOOP
# ---------------------------------------------------------
async def streaming_chat(question, history, session_id):
    if not session_id: return

    # --- INTERCEPT: FAST FETCH ---
    if question.startswith("__FETCH__::"):
        encoded_title = question.split("::")[1]
        readable_title = urllib.parse.unquote(encoded_title)
        
        # Show nice text to user
        display_msg = f"Viewing details for: {readable_title}"
        updated_history = history.copy()
        updated_history.append({"role": "user", "content": display_msg})
        
        # GET DATA INSTANTLY
        response_text = get_cached_details(encoded_title)
        
        updated_history.append({"role": "assistant", "content": response_text})
        yield updated_history
        
        session_handler.save_message(session_id, "user", display_msg)
        session_handler.save_message(session_id, "assistant", response_text)
        return
    # -----------------------------

    # Normal Agent Logic
    session_handler.save_message(session_id, "user", question)
    messages = history.copy()
    messages.append({"role": "user", "content": question})
    clean_message = [{"role": m["role"], "content": m["content"]} for m in messages if "role" in m and "content" in m]

    try:
        response_text = ""
        with trace("Auroville chatbot", trace_id=gen_trace_id()):
            result = Runner.run_streamed(auroville_agent, clean_message)
            async for event in result.stream_events():
                if type(event).__name__ == "RawResponsesStreamEvent":
                    data = event.data
                    if data.__class__.__name__ == "ResponseTextDeltaEvent":
                        response_text += data.delta
                        updated_history = history.copy()
                        updated_history.append({"role": "user", "content": question})
                        updated_history.append({"role": "assistant", "content": response_text})
                        yield updated_history
        
        if response_text:
            session_handler.save_message(session_id, "assistant", response_text)
        else:
            pass # Handle empty

    except Exception as e:
        logger.error(f"Error: {e}")
        yield history + [{"role": "assistant", "content": "An error occurred."}]


# -----------------------------
# GRADIO APP
# -----------------------------
if __name__ == "__main__":
    with gr.Blocks(js=custom_js) as demo:
        gr.Markdown("# 🤖 Auroville Events Chatbot")
        session_id_state = gr.State(value="")
        session_id_bridge = gr.Textbox(visible=False)
        temp_storage_state = gr.State(value="")  
        
        chatbot = gr.Chatbot(height=500, value=[], type='messages')
        
        msg = gr.Textbox(placeholder="Ask about events...", lines=1, show_label=False, elem_id="chat-input")
        
        with gr.Row():
            submit = gr.Button("Send", variant="primary", elem_id="submit-btn")
            clear = gr.Button("Clear")
            new_session_btn = gr.Button("New Session")
        
        session_handler.setup_session_handlers(demo, session_id_state, session_id_bridge, temp_storage_state, chatbot, new_session_btn)        
        
        msg.submit(streaming_chat, inputs=[msg, chatbot, session_id_state], outputs=[chatbot]).then(lambda: "", None, msg)
        submit.click(streaming_chat, inputs=[msg, chatbot, session_id_state], outputs=[chatbot]).then(lambda: "", None, msg)       
        clear.click(lambda: [], None, chatbot)

    logger.info("App started with RAM Caching for Instant Details")
    
    server_port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=server_port, inbrowser=False, debug=False)
