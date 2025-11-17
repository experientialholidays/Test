import gradio as gr
import asyncio
import os
from dotenv import load_dotenv
from agents import Runner, trace, gen_trace_id
from vector_db import VectorDBManager
from db import SessionDBManager
from session_handler import SessionHandler
from auroville_agent import auroville_agent, db_manager, initialize_retriever # <-- IMPORT db_manager and initialize_retriever
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
# db_manager is imported from auroville_agent.py

try:
    print("--- STARTING VECTOR DB INITIALIZATION (FORCE REFRESH=TRUE) ---")
    # 1. Call the initialization method, which returns the vectorstore
    # >>> TEMPORARILY SET TO TRUE FOR THE FIRST DEPLOYMENT <<<
    vectorstore = db_manager.create_or_load_db(force_refresh=True) 
    
    # 2. **CRITICAL NEW STEP:** Initialize the retriever in the agent module
    initialize_retriever(vectorstore) 
    
    print("--- VECTOR DB INITIALIZATION COMPLETE ---")
except Exception as e:
    logger.error(f"FATAL ERROR during DB initialization: {e}")
    # If this fails, the app will still launch but the agent will return an error message.
    pass

# --- END VECTOR DB INITIALIZATION ---


# -----------------------------
# ASYNC STREAMING CHAT FUNCTION
# -----------------------------
async def streaming_chat(question, history, session_id):
    """
    Handle streaming chat using OpenAI Agents SDK with real streaming.
    """
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
                event_type_name = type(event).__name__
                if event_type_name == "RawResponsesStreamEvent":
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

# -----------------------------
# GRADIO APP
# -----------------------------

JS_CODE = """
function attachClickHandlers(msg_input_id, submit_btn_id) {
    const chatbotContainer = document.querySelector('div[data-testid="chatbot"]');
    if (!chatbotContainer) return;

    chatbotContainer.addEventListener('click', function(event) {
        let target = event.target;
        
        if (target.tagName !== 'A') {
            target = target.closest('a');
            if (!target) return;
        }

        const href = target.getAttribute('href');
        if (!href) return;
        
        if (href.startsWith('#TRIGGER_SEARCH::')) {
            event.preventDefault();

            const parts = href.substring(1).split('::');
            if (parts.length < 2) return;
            const query = parts[1]; 
            
            const msgInput = document.getElementById(msg_input_id);
            const submitBtn = document.getElementById(submit_btn_id);

            if (msgInput && submitBtn) {
                msgInput.value = query;
                submitBtn.click();
            }
        }
    });
}
"""

if __name__ == "__main__":
    with gr.Blocks(js=JS_CODE) as demo:
        gr.Markdown("# 🤖 Auroville Events Chatbot")
        
        session_id_state = gr.State(value="")
        session_id_bridge = gr.Textbox(value="", visible=False)
        temp_storage_state = gr.State(value="")  
        
        chatbot = gr.Chatbot(height=500, value=[],type='messages')
        
        msg = gr.Textbox(
            placeholder="Ask me anything about Auroville events...",
            lines=1,
            label="Message",
            show_label=False,
            elem_id="msg_input_field"
        )
        
        with gr.Row():
            submit = gr.Button("Send", variant="primary", elem_id="submit_button")
            clear = gr.Button("Clear Chat")
            new_session_btn = gr.Button("New Session")
        
        demo.load(
            None,
            None,
            None,
            js=f"() => {{ attachClickHandlers('msg_input_field', 'submit_button'); }}"
        )
        
        session_handler.setup_session_handlers(
            demo=demo,
            session_id_state=session_id_state,
            session_id_bridge=session_id_bridge,
            temp_storage_state=temp_storage_state,
            chatbot=chatbot,
            new_session_btn=new_session_btn
        )        
        
        msg.submit(streaming_chat,inputs=[msg, chatbot, session_id_state],outputs=[chatbot]).then(lambda: "",None,msg )
        submit.click(streaming_chat,inputs=[msg, chatbot, session_id_state],outputs=[chatbot]).then(lambda: "",None,msg)       
        
        clear.click(lambda: [], None, chatbot)

    logger.info("App started with OpenAI Agents SDK - Real Streaming Enabled")
    
    server_port = int(os.environ.get("PORT", 8080))
    server_host = "0.0.0.0"
    
    demo.launch(server_name=server_host, server_port=server_port, inbrowser=False, debug=False)
