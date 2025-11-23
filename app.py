import gradio as gr
import asyncio
import os
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
# ASYNC STREAMING CHAT FUNCTION
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
                        updated_history.append({"role": "user", "content": question})
                        updated_history.append({"role": "assistant", "content": response_text})
                        
                        yield updated_history
            
        if response_text:
            session_handler.save_message(session_id, "assistant", response_text)
        else:
            error_msg = "I apologize, but I couldn't generate a proper response. Please try again."
            updated_history = history.copy()
            updated_history.append({"role": "user", "content": question})
            updated_history.append({"role": "assistant", "content": error_msg})
            yield updated_history
            session_handler.save_message(session_id, "assistant", error_msg)
        
    except Exception as e:
        error_msg = f"I encountered an error: {str(e)}. Please try again."
        logger.error(f"Error: {e}")
        
        updated_history = history.copy()
        updated_history.append({"role": "user", "content": question})
        updated_history.append({"role": "assistant", "content": error_msg})
        
        yield updated_history
        session_handler.save_message(session_id, "assistant", error_msg)


# ----------------------------------------------------------
# UPDATED JS
# ----------------------------------------------------------

JS_CODE = """
function attachClickHandlers(msg_input_id, submit_btn_id) {

    function fillAndSend(text) {
        const msgInput = document.getElementById(msg_input_id);
        const submitBtn = document.getElementById(submit_btn_id);

        if (msgInput && submitBtn) {
            msgInput.value = text;
            msgInput.dispatchEvent(new Event('input', { bubbles: true }));
            submitBtn.click();
        }
    }

    document.addEventListener('click', function(event) {

        const anchor = event.target.closest &&
            event.target.closest('a[href^="#DETAILS::"], a[href^="#SHOWDAILY::"], a[href^="#FETCH::"]');

        if (!anchor) return;

        const href = anchor.getAttribute('href');
        if (!href) return;

        event.preventDefault();
        event.stopPropagation();

        // DETAILS
        if (href.startsWith("#DETAILS::")) {
            const parts = href.substring(1).split("::");
            const match = parts[1].match(/(\\d+)/);
            if (match) {
                fillAndSend("details(" + match[1] + ")");
            }
            return;
        }

        // SHOW DAILY EVENTS
        if (href.startsWith("#SHOWDAILY::")) {
            const parts = href.substring(1).split("::");
            const choice = parts[1];

            if (choice === "YES") {
                fillAndSend("show daily events");
            } else if (choice === "NO") {
                fillAndSend("no");
            }
            return;
        }

        // FETCH
        if (href.startsWith("#FETCH::")) {
            const parts = href.substring(1).split("::");
            fillAndSend("#FETCH::" + parts[1]);
            return;
        }
    });
}
"""


# ----------------------------------------------------------
# GRADIO APP
# ----------------------------------------------------------

if __name__ == "__main__":

    with gr.Blocks() as demo:

        # ✅ FIX APPLIED: JS now executes
        gr.HTML(f"<script>{JS_CODE}</script>")

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
            js="() => { attachClickHandlers('msg_input_field', 'submit_button'); }"
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

    logger.info("Auroville App Started with Updated Click Support")

    server_port = int(os.environ.get("PORT", 8080))
    server_host = "0.0.0.0"

    demo.launch(
        server_name=server_host,
        server_port=server_port,
        inbrowser=False,
        debug=False
    )
