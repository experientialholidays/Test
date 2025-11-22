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

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

SESSION_DB_FILE = "sessions.db"
session_db_manager = SessionDBManager(db_file=SESSION_DB_FILE)
session_handler = SessionHandler(session_db_manager=session_db_manager)

try:
    vectorstore = db_manager.create_or_load_db(force_refresh=False)
    initialize_retriever(vectorstore)
except:
    pass


# ---------------------------
#  FIXED JAVASCRIPT LOADER
# ---------------------------

def js_file():
    return """
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

            if (href.startsWith("#DETAILS::")) {
                const parts = href.substring(1).split("::");
                const match = parts[1].match(/(\\d+)/);
                if (match) fillAndSend("details(" + match[1] + ")");
                return;
            }

            if (href.startsWith("#SHOWDAILY::")) {
                const parts = href.substring(1).split("::");
                const choice = parts[1];
                if (choice === "YES") fillAndSend("show daily events");
                else if (choice === "NO") fillAndSend("no");
                return;
            }

            if (href.startsWith("#FETCH::")) {
                const parts = href.substring(1).split("::");
                fillAndSend("#FETCH::" + parts[1]);
                return;
            }
        });
    }
    """


# ----------------------------------------------------------
# STREAMING CHAT
# ----------------------------------------------------------

async def streaming_chat(question, history, session_id):
    session_handler.save_message(session_id, "user", question)

    msgs = history.copy()
    msgs.append({"role": "user", "content": question})

    response_text = ""

    try:
        clean_message = [{"role": m["role"], "content": m["content"]} for m in msgs]

        trace_id = gen_trace_id()
        with trace("Auroville chatbot", trace_id=trace_id):

            result = Runner.run_streamed(auroville_agent, clean_message)

            async for event in result.stream_events():
                if type(event).__name__ == "RawResponsesStreamEvent":
                    delta = event.data
                    if delta.__class__.__name__ == "ResponseTextDeltaEvent":
                        response_text += delta.delta
                        out = history.copy()
                        out.append({"role": "assistant", "content": response_text})
                        yield out

        session_handler.save_message(session_id, "assistant", response_text)

    except Exception as e:
        err = f"I encountered an error: {str(e)}"
        out = history.copy()
        out.append({"role": "assistant", "content": err})
        yield out


# ----------------------------------------------------------
#  UI
# ----------------------------------------------------------

with gr.Blocks(js=js_file()) as demo:

    gr.Markdown("# 🤖 Auroville Events Chatbot")

    session_id_state = gr.State("")
    session_id_bridge = gr.Textbox(value="", visible=False)
    temp_storage_state = gr.State("")

    chatbot = gr.Chatbot(height=500, value=[])

    msg = gr.Textbox(
        placeholder="Ask me anything about Auroville events...",
        lines=1,
        show_label=False,
        elem_id="msg_input_field"
    )

    with gr.Row():
        submit = gr.Button("Send", variant="primary", elem_id="submit_button")
        new_session_btn = gr.Button("New Session")

    demo.load(
        None, None, None,
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

    msg.submit(streaming_chat, [msg, chatbot, session_id_state], [chatbot]).then(
        lambda: "", None, msg
    )

    submit.click(streaming_chat, [msg, chatbot, session_id_state], [chatbot]).then(
        lambda: "", None, msg
    )


demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
