import os
import logging
import asyncio
import gradio as gr
# Import the agent and data store from your auroville_agent module
from auroville_agent import auroville_agent, EVENT_DATA_STORE, format_event_card

# --- 1. Setup & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Gradio Interface Function ---

async def agent_query_handler(user_input: str) -> str:
    """
    Handles user input, calls the async agent, and manages dynamic clicks.
    """
    if not user_input.strip():
        return "Please enter a question to search for events."

    # --- Handle #FETCH:: Click Logic ---
    if user_input.startswith("#FETCH::"):
        try:
            # Extract the safe_key from the command, e.g., #FETCH::Event%20Name
            safe_key = user_input.split("::")[1]
            doc = EVENT_DATA_STORE.get(safe_key)
            
            if doc:
                # Use your existing formatting helper to return the full card
                return format_event_card(doc.metadata, doc.page_content)
            else:
                return "Error: Event details not found in the cache. Please run the search again."
        except Exception as e:
            return f"Error processing fetch request: {e}"

    # --- Handle New Search Query ---
    try:
         response_text = await auroville_agent.ainvoke(
         {"user_input": user_input, "session_id": "gradio_session"}
         )
        return response_text

    except RuntimeError as e:
        # This catches the RuntimeError raised by get_initialized_retriever()
        if "Retriever initialization failed" in str(e):
             return "🚨 System Error: The event database failed to load during the request. This may be due to a Cloud Run startup timeout. Please try again in 30 seconds."
        return f"An unexpected error occurred during agent execution: {e}"
        
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- 3. Gradio Blocks Setup ---

with gr.Blocks(title="Auroville Events Assistant") as demo:
    gr.Markdown("# 📅 Auroville Events Assistant")
    gr.Markdown(
        "Ask a question about events, workshops, or activities (e.g., 'What events are happening tomorrow in Youth Centre?').\n\n"
        "**NOTE:** If the first search is slow, it is loading the database. Subsequent searches will be faster."
    )

    query_box = gr.Textbox(
        label="Your Query", 
        placeholder="e.g., What is happening this weekend?", 
        lines=2
    )
    
    # Assign an ID to the output box for the JavaScript observer to target reliably
    output_box = gr.Markdown(
        label="Event Search Results", 
        value="Results will appear here. Click on an event name for full details.",
        elem_id="output_box_id"
    )

    submit_btn = gr.Button("Search Events", variant="primary")

    submit_btn.click(
        fn=agent_query_handler,
        inputs=[query_box],
        outputs=[output_box],
        queue=True
    )
    
    # --- Interactivity Hook for Fetching Details ---
    
    # 1. Hidden input that receives the custom markdown link content
    fetch_trigger = gr.Textbox(visible=False, label="Fetch Trigger", elem_id="fetch_trigger_id")

    # 2. Hidden button that listens for the JS trigger to execute the fetch logic
    fetch_hidden_btn = gr.Button("Fetch Details", elem_id="fetch_hidden_btn", visible=False)

    # 3. Hidden button's click handler runs the agent_query_handler with the FETCH command
    fetch_hidden_btn.click(
        fn=agent_query_handler,
        inputs=[fetch_trigger],
        outputs=[output_box],
        queue=False 
    )
    
    # 4. Global JS Injection using gr.HTML (Workaround for _js TypeError)
    # This code block injects the JavaScript observer globally using the HTML component.
    js_code = """
    <script>
        function attachFetchListeners() {
            const outputElement = document.getElementById('output_box_id');
            if (outputElement) {
                // Find all links that start with #FETCH::
                outputElement.querySelectorAll('a[href^="#FETCH::"]').forEach(link => {
                    // Prevent multiple listeners if the function is called multiple times
                    if (!link.hasAttribute('data-fetch-listener')) {
                        link.setAttribute('data-fetch-listener', 'true');
                        link.onclick = function(e) {
                            e.preventDefault();
                            const fetchCommand = this.getAttribute('href').substring(1);
                            
                            // Target the textarea inside the hidden fetch_trigger box
                            const triggerElement = document.getElementById('fetch_trigger_id').querySelector('textarea');
                            if (triggerElement) {
                                triggerElement.value = fetchCommand;
                                
                                // Click the hidden button to trigger the fetch_details logic
                                document.getElementById('fetch_hidden_btn').click();
                            }
                        };
                    }
                });
            }
        }

        // Use a MutationObserver to re-attach the listeners whenever the output content changes
        const observerTarget = document.getElementById('output_box_id');
        if (observerTarget) {
            const observer = new MutationObserver(attachFetchListeners);
            // Observe changes to the content and subtree
            observer.observe(observerTarget, { childList: true, subtree: true });
        }
        
        // Also run on initial load
        attachFetchListeners();
    </script>
    """
    # Insert the JS code as raw HTML at the end of the demo
    gr.HTML(js_code)


# --- 4. Launch Settings for Cloud Run ---

if __name__ == "__main__":
    # Cloud Run uses the PORT environment variable
    port = int(os.environ.get("PORT", 7860))
    
    # Set the server name to '0.0.0.0' to listen on all interfaces (required for Docker/Cloud Run)
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        # Setting a concurrency limit can sometimes help manage resources on Cloud Run
        max_threads=20 
    )
