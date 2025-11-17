# app.py - Standard Cloud Run/Web Server Entry Point

import os
import json
import logging
from flask import Flask, request, jsonify, make_response
from auroville_agent import auroville_agent, EVENT_DATA_STORE # Import the agent and data store

# Set up logging for the entry point
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask application setup
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """Simple check for service health."""
    return "Auroville Events Assistant Service is running.", 200

@app.route('/query', methods=['POST'])
async def handle_query():
    """
    Handles incoming user queries and processes them through the Auroville Agent.
    """
    try:
        data = request.get_json(silent=True)
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field in request body."}), 400

        user_query = data['query']
        session_id = data.get('session_id', 'default_session') # Use session_id for context management if needed
        
        logger.info(f"Received query for session {session_id}: {user_query}")

        # Execute the agent logic asynchronously
        response_text = await auroville_agent.arun(
            user_input=user_query,
            session_id=session_id
        )

        return jsonify({"response": response_text}), 200

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        # Catch and handle potential initialization errors here if the guarded lazy load fails
        if "Retriever initialization failed" in str(e):
             return jsonify({"error": "System Error: The event database is unavailable. Please try again later."}), 503
        
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/fetch_details', methods=['POST'])
def fetch_details():
    """
    Retrieves full event details based on a safe key from the EVENT_DATA_STORE.
    """
    try:
        data = request.get_json(silent=True)
        if not data or 'key' not in data:
            return jsonify({"error": "Missing 'key' field in request body."}), 400

        safe_key = data['key']
        doc = EVENT_DATA_STORE.get(safe_key)

        if doc:
            # Assuming you want to return the full document page content and metadata
            return jsonify({
                "content": doc.page_content,
                "metadata": doc.metadata
            }), 200
        else:
            return jsonify({"error": "Event key not found."}), 404

    except Exception as e:
        logger.error(f"Error fetching details: {e}")
        return jsonify({"error": "An internal server error occurred during detail retrieval."}), 500


if __name__ == '__main__':
    # Cloud Run/Gunicorn will set the PORT environment variable
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, threaded=False)

