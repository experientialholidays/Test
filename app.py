Import gradio as gr
import asyncio
import os
from dotenv import load_dotenv
from agents import Runner, trace, gen_trace_id
from vector_db import VectorDBManager
from db import SessionDBManager
from session_handler import SessionHandler
from auroville_agent import auroville_agent
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
# NOTE: This block is the SLOW point that causes the Cloud Run failure.
# We are setting force_refresh=True TEMPORARILY to force a rebuild with the new headers.
# *** YOU MUST CHANGE THIS BACK TO False AFTER THE FIRST SUCCESSFUL DEPLOYMENT ***
VECTOR_DB_NAME = "vector_db"
DB_FOLDER = "input"
db_manager = VectorDBManager(folder=DB_FOLDER, db_name=VECTOR_DB_NAME)

try:
    print("--- STARTING VECTOR DB INITIALIZATION (FORCE REFRESH=TRUE) ---")
    # >>> TEMPORARILY SET TO TRUE FOR THE FIRST DEPLOYMENT <<<
    vectorstore = db_manager.create_or_load_db(force_refresh=True) 
    print("--- VECTOR DB INITIALIZATION COMPLETE ---")
except Exception as e:
    logger.error(f"FATAL ERROR during DB initialization: {e}")
    # In a real app, you might exit here, but for Cloud Run, we proceed to launch the app
    # to let the error be captured by Gradio or Cloud Run logging.
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
    
    # Save user message
    session_handler.save_message(session_id, "user", question)
    
    # Build conversation history for agent
    messages = history.copy() 
    messages.append({"role": "user", "content": question})
    
    try:
        response_text = ""
        # Clean the history to keep only 'role' and 'content'
        clean_message = [{"role": m["role"], "content": m["content"]} for m in messages if "role" in m and "content" in m]

        # Stream using Agent directly
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
            
        # Save assistant response
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

# --- CUSTOM JAVASCRIPT FOR CLICK-BASED SEARCH ---
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
        
        // Check for the special link marker
        if (href.startsWith('#TRIGGER_SEARCH::')) {
            event.preventDefault();

            // 1. Extract the query: '#TRIGGER_SEARCH::query::specificity'
            const parts = href.substring(1).split('::');
            if (parts.length < 2) return;
            const query = parts[1]; 
            
            // 2. Get the Gradio message input and submit button
            const msgInput = document.getElementById(msg_input_id);
            const submitBtn = document.getElementById(submit_btn_id);

            if (msgInput && submitBtn) {
                // 3. Set the query into the textbox
                msgInput.value = query;

                // 4. Programmatically trigger the click
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
