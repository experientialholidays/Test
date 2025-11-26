import gradio as gr
import asyncio
import os
from dotenv import load_dotenv
from agents import Runner, trace, gen_trace_id
from vector_db import VectorDBManager
from db import SessionDBManager
from session_handler import SessionHandler

# NEW FIX: Import the functions using aliases (fetch_...) to avoid a conflict
# with non-callable FunctionTool objects that might be named the same.
from auroville_agent import (
    auroville_agent, 
    db_manager, 
    initialize_retriever, 
    get_event_details as fetch_event_details, # Renamed for direct calling
    get_daily_events as fetch_daily_events,   # Renamed for direct calling
    EVENT_DATA_STORE
)

import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

SESSION_DB_FILE = "sessions.db"
session_db_manager = SessionDBManager(db_file=SESSION_DB_FILE)
session_handler = SessionHandler(session_db_manager=session_db_manager)

# --- VECTOR DB INITIALIZATION ---
VECTOR_DB_NAME = "vector_db"
DB_FOLDER = "input"

try:
    print("--- STARTING VECTOR DB INITIALIZATION (FORCE REFRESH=FALSE) ---")
    vectorstore = db_manager.create_or_load_db(force_refresh=False)
    initialize_retriever(vectorstore)
    print("--- VECTOR DB INITIALIZATION COMPLETE ---")
except Exception as e:
    logger.error(f"FATAL ERROR during DB initialization: {e}")
    pass

# ----------------------------------------------------------
# Helper: parse direct commands
# ----------------------------------------------------------
DETAILS_RE = re.compile(r'^\s*details\(\s*(\d+)\s*\)\s*$', re.IGNORECASE)
DIGITS_RE = re.compile(r'^\s*(\d+)\s*$')
SHOW_DAILY_ALIASES = {
    "show daily events",
    "show daily event",
    "show daily",
    "show daily events please",
    "show daily events, please",
    "show daily events yes",
    "show daily events y",
}

# ----------------------------------------------------------
# ASYNC STREAMING CHAT FUNCTION (with routing override)
# ----------------------------------------------------------

async def streaming_chat(question, history, session_id):
    """
    This function now performs quick routing for *direct* commands:
    FIXED: Uses fetch_event_details and fetch_daily_events to avoid TypeError.
    """

    # --- INPUT SANITIZATION FIX ---
    if isinstance(question, dict):
        q_raw = question.get('text', "")
    else:
        q_raw = question or ""
    
    q = q_raw.strip()

    logger.info(f'Processing chat for session: {session_id} | question: {q!r}')
    if not session_id or session_id == "null":
        logger.info("ERROR: No valid session_id!")
        return

    # 1) Direct details(...) pattern
    m = DETAILS_RE.match(q)
    if m:
        idx = int(m.group(1))
        logger.info(f"Routing to get_event_details for id={idx} (direct details() input).")
        # FIXED CALL: Using the callable alias
        result = fetch_event_details(f"details({idx})")
        # Save to session and return a one-shot response
        session_handler.save_message(session_id, "user", q)
        session_handler.save_message(session_id, "assistant", result)
        updated_history = history.copy()
        updated_history.append({"role": "user", "content": q})
        updated_history.append({"role": "assistant", "content": result})
        yield updated_history
        return

    # 2) Plain integer e.g., "4"
    m2 = DIGITS_RE.match(q)
    if m2:
        idx = int(m2.group(1))
        logger.info(f"Routing to get_event_details for id={idx} (plain integer input).")
        # FIXED CALL: Using the callable alias
        result = fetch_event_details(str(idx))
        session_handler.save_message(session_id, "user", q)
        session_handler.save_message(session_id, "assistant", result)
        updated_history = history.copy()
        updated_history.append({"role": "user", "content": q})
        updated_history.append({"role": "assistant", "content": result})
        yield updated_history
        return

    # 3) Show daily events (exact phrases used by the JS)
    if q.lower() in SHOW_DAILY_ALIASES:
        # Determine start index for daily events numbering (length of existing cache)
        try:
            last_index = max(EVENT_DATA_STORE.keys()) if EVENT_DATA_STORE else 0
        except Exception:
            last_index = 0
        logger.info(f"Routing to get_daily_events(start_number={last_index}).")
        # FIXED CALL: Using the callable alias
        result = fetch_daily_events(start_number=last_index)
        session_handler.save_message(session_id, "user", q)
        session_handler.save_message(session_id, "assistant", result)
        updated_history = history.copy()
        updated_history.append({"role": "user", "content": q})
        updated_history.append({"role": "assistant", "content": result})
        yield updated_history
        return

    # 4) Otherwise — fall back to LLM-driven flow (streaming)
    session_handler.save_message(session_id, "user", q)
    messages = history.copy()
    messages.append({"role": "user", "content": q})

    try:
        response_text = ""
        clean_message = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if "role" in m and "content" in m
        ]

        trace_id = gen_trace_id()
        with trace("Auroville chatbot", trace_id=trace_id):
            logger.info(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")

            result = Runner.run_streamed(auroville_agent, clean_message)

            async for event in result.stream_events():
                if type(event).__name__ == "RawResponsesStreamEvent":
                    data = event.data
                    if data.__class__.__name__ == "ResponseTextDeltaEvent":
                        response_text += data.delta

                        updated_history = history.copy()
                        updated_history.append({"role": "user", "content": q})
                        updated_history.append({"role": "assistant", "content": response_text})

                        yield updated_history

        if response_text:
            session_handler.save_message(session_id, "assistant", response_text)
        else:
            error_msg = "I apologize, but I couldn't generate a proper response. Please try again."
            updated_history = history.copy()
            updated_history.append({"role": "user", "content": q})
            updated_history.append({"role": "assistant", "content": error_msg})
            yield updated_history
            session_handler.save_message(session_id, "assistant", error_msg)

    except Exception as e:
        error_msg = f"I encountered an error: {str(e)}. Please try again."
        logger.error(f"Error: {e}")

        updated_history = history.copy()
        updated_history.append({"role": "user", "content": q})
        updated_history.append({"role": "assistant", "content": error_msg})

        yield updated_history
        session_handler.save_message(session_id, "assistant", error_msg)


# ----------------------------------------------------------
# UPDATED JS - FIXES CLICK HANDLERS AND BLANK MESSAGE ISSUE
# ----------------------------------------------------------

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
        if (!href) return;

        const submitBtn = document.getElementById(submit_btn_id);
        
        // Use querySelector to find the textarea inside the container.
        const inputContainer = document.getElementById(msg_input_id);
        const msgInput = inputContainer ? inputContainer.querySelector('textarea') : null;

        if (!msgInput || !submitBtn) {
            console.error("Gradio elements not found.");
            return;
        }

        let textToSend = "";

        // --- DETAILS ---
        if (href.startsWith("#DETAILS::")) {
            const parts = href.substring(1).split("::");
            const match = parts[1].match(/(\\d+)/);
            if (match) {
                textToSend = "details(" + match[1] + ")";
            }
        }

        // --- SHOW DAILY ---
        else if (href.startsWith("#SHOWDAILY::")) {
            const parts = href.substring(1).split("::");
            const choice = parts[1];
            if (choice === "YES") {
                textToSend = "show daily events";
            } else if (choice === "NO") {
                textToSend = "no";
            }
        }

        // 4. Fill and Submit
        if (textToSend) {
            // 1. Set the value
            msgInput.value = textToSend;
            
            // 2. Dispatch the input event to tell Gradio/React the value changed
            msgInput.dispatchEvent(new Event('input', { bubbles: true }));

            // 3. Click Submit (Increased delay to ensure state is registered)
            setTimeout(() => {
                submitBtn.click();
            }, 200); 
        }
    });
}
"""

# ----------------------------------------------------------
# GRADIO APP
# ----------------------------------------------------------

if __name__ == "__main__":

    with gr.Blocks() as demo:

         gr.Markdown("# 🤖 Auroville Events Chatbot")

         session_id_state = gr.State("")
         session_id_bridge = gr.Textbox(value="", visible=False)
         temp_storage_state = gr.State("")

         chatbot = gr.Chatbot(height=500, value=[])

         msg = gr.Textbox(
            placeholder="Ask me anything about Auroville events...",
            lines=1,
            label="Message",
            show_label=False,
            elem_id="msg_input_field"
         )

         with gr.Row():
            submit = gr.Button("Send", variant="primary", elem_id="submit_button")
            new_session_btn = gr.Button("New Session")

         demo.load(
             None,
             None,
             None,
             js=f"() => {{ {JS_CODE} attachClickHandlers('msg_input_field', 'submit_button'); }}"
         )

         session_handler.setup_session_handlers(
             demo=demo,
             session_id_state=session_id_state,
             session_id_bridge=session_id_bridge,
             temp_storage_state=temp_storage_state,
             chatbot=chatbot,
             new_session_btn=new_session_btn
         )

         msg.submit(
             streaming_chat,
             inputs=[msg, chatbot, session_id_state],
             outputs=[chatbot]
         ).then(lambda: "", None, msg)

         submit.click(
             streaming_chat,
             inputs=[msg, chatbot, session_id_state],
             outputs=[chatbot]
         ).then(lambda: "", None, msg)

    logger.info("Auroville App Started with Fixed Click & Input Handlers")

    server_port = int(os.environ.get("PORT", 8080))
    server_host = "0.0.0.0"

    demo.launch(
        server_name=server_host,
        server_port=server_port,
        inbrowser=False,
        debug=False
    )
