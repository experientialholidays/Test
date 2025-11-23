import gradio as gr
import asyncio
import os
import re
from dotenv import load_dotenv
from agents import Runner, trace, gen_trace_id
from vector_db import VectorDBManager
from db import SessionDBManager
from session_handler import SessionHandler
from auroville_agent import auroville_agent, db_manager, initialize_retriever
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

SESSION_DB_FILE = "sessions.db"
session_db_manager = SessionDBManager(db_file=SESSION_DB_FILE)
session_handler = SessionHandler(session_db_manager=session_db_manager)

# --- VECTOR DB INITIALIZATION ---
try:
    logger.info("--- STARTING VECTOR DB INITIALIZATION (FORCE REFRESH=FALSE) ---")
    vectorstore = db_manager.create_or_load_db(force_refresh=False)
    initialize_retriever(vectorstore)
    logger.info("--- VECTOR DB INITIALIZATION COMPLETE ---")
except Exception as e:
    logger.error(f"FATAL ERROR during DB initialization: {e}")
    pass


# ----------------------------------------------------------
# Streaming chat (uses your existing agent)
# ----------------------------------------------------------

async def streaming_chat(question, history, session_id):
    logger.info(f'Processing chat for session: {session_id}')

    if not session_id or session_id == "null":
        logger.info("ERROR: No valid session_id!")
        return

    session_handler.save_message(session_id, "user", question)
    messages = history.copy()
    messages.append({"role": "user", "content": question})

    try:
        response_text = ""
        clean_message = [{"role": m["role"], "content": m["content"]} for m in messages if "role" in m and "content" in m]

        trace_id = gen_trace_id()
        with trace("Auroville chatbot", trace_id=trace_id):
            logger.info(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")

            result = Runner.run_streamed(auroville_agent, clean_message)

            async for event in result.stream_events():
                # stream deltas to the chat UI
                if type(event).__name__ == "RawResponsesStreamEvent":
                    data = event.data
                    if data.__class__.__name__ == "ResponseTextDeltaEvent":
                        response_text += data.delta
                        updated_history = history.copy()
                        updated_history.append({"role": "user", "content": question})
                        updated_history.append({"role": "assistant", "content": response_text})
                        yield updated_history

        # final save
        if response_text:
            session_handler.save_message(session_id, "assistant", response_text)
    except Exception as e:
        error_msg = f"I encountered an error: {str(e)}"
        logger.error(error_msg)
        updated_history = history.copy()
        updated_history.append({"role": "assistant", "content": error_msg})
        yield updated_history
        session_handler.save_message(session_id, "assistant", error_msg)


# ----------------------------------------------------------
# Process assistant message and render real buttons
# ----------------------------------------------------------
# This function examines the last assistant message (from the chatbot value),
# finds markdown links of the form [Label](#DETAILS::123) or [Label](#FETCH::key)
# and returns an HTML fragment (buttons + JS) that will be placed under the chat.
#
# Buttons call JS that fills the input and auto-clicks "Send", triggering the agent.
# ----------------------------------------------------------

def process_buttons(chatbot_value):
    """
    chatbot_value: list of message tuples [[user, assistant], ...] as returned by the chatbot component
    returns: HTML string (sanitized=False when added to gr.HTML) containing buttons and JS
    """
    try:
        if not chatbot_value:
            return ""  # clear buttons
        last = chatbot_value[-1]
        # last can be tuple (user, assistant) or dict depending on gradio version
        assistant_text = ""
        if isinstance(last, (list, tuple)) and len(last) >= 2:
            assistant_text = last[1] or ""
        elif isinstance(last, dict) and "content" in last:
            assistant_text = last.get("content", "")
        else:
            assistant_text = str(last)

        if not assistant_text:
            return ""

        # Find markdown links like [Label](#DETAILS::123) or [Label](#FETCH::key)
        pattern = re.compile(r'\[([^\]]+)\]\(#(DETAILS|FETCH)::([^\)]+)\)', re.IGNORECASE)
        matches = pattern.findall(assistant_text)

        if not matches:
            # nothing to render; clear previous buttons
            return ""

        # Build HTML for buttons
        parts = []
        parts.append("<div class='event-buttons-container' style='margin-top:8px;margin-bottom:12px;'>")
        for idx, (label, typ, key) in enumerate(matches, start=1):
            safe_label = label.replace('"', '&quot;')
            safe_key = key.replace('"', '&quot;')
            typ_up = typ.upper()
            # Unique id for button
            btn_id = f"event_btn_{idx}"
            parts.append(
                f"<button id='{btn_id}' class='event-btn' data-typ='{typ_up}' data-key='{safe_key}' "
                "style='margin:4px;padding:8px 12px;border-radius:6px;border:1px solid #ccc;background:#f7f7f7;cursor:pointer;'>"
                f"{safe_label}</button>"
            )
        parts.append("</div>")

        # Add JS that wires the buttons to fill & send the input
        # Uses existing input elem_id 'msg_input_field' and submit button id 'submit_button'
        script = r"""
<script>
(function(){
  function wireButtons(){
    const btns = document.querySelectorAll('.event-btn');
    const input = document.getElementById('msg_input_field');
    const send = document.getElementById('submit_button');
    if (!input || !send) return;
    btns.forEach(b=>{
      // prevent double-binding
      if (b.dataset.bound === '1') return;
      b.dataset.bound='1';
      b.addEventListener('click', function(e){
        e.preventDefault();
        const typ = b.dataset.typ;
        const key = b.dataset.key;
        if (typ === 'DETAILS'){
          // key might be numeric or a small id; pass details(key)
          input.value = 'details(' + key + ')';
        } else if (typ === 'FETCH'){
          input.value = '#FETCH::' + key;
        } else {
          input.value = key;
        }
        // notify React/Gradio of change and click send
        input.dispatchEvent(new Event('input', {bubbles:true}));
        send.click();
      });
    });
  }
  // run once, and again after a small delay (handles re-rendering)
  setTimeout(wireButtons, 50);
  setTimeout(wireButtons, 500);
})();
</script>
"""
        parts.append(script)
        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error in process_buttons: {e}")
        return ""


# ----------------------------------------------------------
# GRADIO 4.x UI
# ----------------------------------------------------------

def build_ui():
    # custom JS loader — kept minimal (not required for button logic, but can run any global js)
    custom_js = gr.JS(
        """
        // placeholder for any global JS you might want to run
        // event wiring is handled in the HTML returned by process_buttons
        """
    )

    with gr.Blocks() as demo:
        # run the JS (no-op here but ensures gr.JS is loaded)
        custom_js.run()

        gr.Markdown("# 🤖 Auroville Events Chatbot")

        session_id_state = gr.State("")
        session_id_bridge = gr.Textbox(value="", visible=False)
        temp_storage_state = gr.State("")

        chatbot = gr.Chatbot(elem_classes="auro-chatbox", height=500, value=[])

        msg = gr.Textbox(
            placeholder="Ask me anything about Auroville events...",
            lines=1,
            show_label=False,
            elem_id="msg_input_field"
        )

        with gr.Row():
            submit = gr.Button("Send", variant="primary", elem_id="submit_button")
            new_session_btn = gr.Button("New Session")

        # This HTML block will be updated to contain the generated buttons (sanitization disabled)
        buttons_html = gr.HTML(value="", elem_id="event_buttons", visible=True)

        # Wire up handlers: streaming_chat updates chatbot; when chat update completes, we call process_buttons
        # to populate the buttons_html component with real buttons for the last assistant message.

        # For the submit via Enter
        msg.submit(streaming_chat, inputs=[msg, chatbot, session_id_state], outputs=[chatbot]) \
            .then(process_buttons, inputs=[chatbot], outputs=[buttons_html]) \
            .then(lambda: "", None, msg)  # clear input

        # For explicit send button
        submit.click(streaming_chat, inputs=[msg, chatbot, session_id_state], outputs=[chatbot]) \
            .then(process_buttons, inputs=[chatbot], outputs=[buttons_html]) \
            .then(lambda: "", None, msg)

        # Setup session handlers as before
        session_handler.setup_session_handlers(
            demo=demo,
            session_id_state=session_id_state,
            session_id_bridge=session_id_bridge,
            temp_storage_state=temp_storage_state,
            chatbot=chatbot,
            new_session_btn=new_session_btn
        )

        return demo

# Build and launch
if __name__ == "__main__":
    demo = build_ui()
    logger.info("Launching upgraded Gradio 4.x Auroville app (buttons enabled)")
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
