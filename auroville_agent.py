import os
from datetime import datetime
from agents import Agent, function_tool
from vector_db import VectorDBManager
from vectordb_query_selector_agent import vectordb_query_selector_agent
from vectordb_filtering_agent import vectordb_filtering_agent
from openai import AsyncOpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
MODEL = "gpt-4.1-mini"
DB_FOLDER = "input"
VECTOR_DB_NAME = "vector_db"

# Vector Database
db_manager = VectorDBManager(folder=DB_FOLDER, db_name=VECTOR_DB_NAME)
vectorstore = db_manager.create_or_load_db(force_refresh=False)
retriever = db_manager.get_retriever(k=50)

INSTRUCTIONS = f"""You are a helpful AI assistant for Auroville events and activities.
Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.

Your role is to help users find information about events, activities, workshops, and schedules in Auroville.

You have access to three tools:
1) `vectordb_query_selector_agent` - Generates the best possible refined search query and specificity based on the user input.
2) `search_auroville_events` - Retrieves relevant events from the vector database.
3) `vectordb_filtering_agent` - Filters, organizes, and presents the final information to the user clearly.

**CRITICAL WORKFLOW - Follow this exact sequence:**

Step 1: Call `vectordb_query_selector_agent` with the user's question
   - This will return a refined search query and specificity level

Step 2: Call `search_auroville_events` using:
   - search_query: The refined query from Step 1
   - specificity: The specificity level from Step 1

Step 3: Call `vectordb_filtering_agent` with:
   - user_query: The original user question
   - raw_results: The COMPLETE output from `search_auroville_events` in Step 2
   
   **IMPORTANT**: You MUST pass the ENTIRE search results text to the filtering agent.
   The filtering agent needs the raw data to filter and format it properly.

Step 4: Return the filtering agent's response to the user

**Example:**
User: "What's happening today?"

1. Call vectordb_query_selector_agent("What's happening today?")
   → Returns: {{"search_query": "events in Auroville on October 12, 2025", "specificity": "Broad"}}

2. Call search_auroville_events(search_query="events in Auroville on October 12, 2025", specificity="Broad")
   → Returns: "Here is relevant information about Auroville events:\n\nDocument 1: Dance workshop...\nDocument 2: Yoga class..."

3. Call vectordb_filtering_agent(
     user_query="What's happening today?",
     raw_results="Here is relevant information about Auroville events:\n\nDocument 1: Dance workshop...\nDocument 2: Yoga class..."
   )
   → Returns: Filtered and formatted response

4. Return that filtered response to the user

If the user asks about something unrelated to events, reply conversationally without using tools.
"""

@function_tool
def search_auroville_events(search_query: str,specificity: str = "Broad") -> str:
    """
    Search for information about Auroville events and activities. 
    Use this tool whenever the user asks about events, activities, schedules, or anything related to Auroville.
    
    Args:
        search_query: The search query about Auroville events (e.g., 'dance events in October', 'music workshops', 'yoga classes')
        specificity: Determine query specificity:
                    - Broad (general date/day queries)
                    - Specific (particular event/activity queries)
        
    Returns:
        str: Relevant information about Auroville events
    """
    logger.info(f"RAG Tool called with query: {search_query}")
    
    # Dynamically adjust retrieval depth
    k_value = 50 if specificity.lower() == "broad" else 10
    # Retrieve relevant documents (uses k=50 from retriever config)
    docs = retriever.get_relevant_documents(search_query,k=k_value)

    
    if not docs:
        return "No relevant information found about Auroville events."
    
    # Format all retrieved documents
    context = "\n\n".join([
        f"Document {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])
    
    logger.info(f"Retrieved {len(docs)} documents for RAG context")
    
    return f"Here is relevant information about Auroville events:\n\n{context}"


tools = [search_auroville_events,
         vectordb_query_selector_agent.as_tool(tool_name="vectordb_query_selector_agent", tool_description="Generates a input query for the vector db search"),
         vectordb_filtering_agent.as_tool(tool_name="vectordb_filtering_agent", tool_description="Filters and formats raw vector database results for the user. Requires 'user_query' (original question) and 'raw_results' (complete search output) as inputs.")]
# -----------------------------
# CREATE AGENT
# -----------------------------
auroville_agent = Agent(
    name="Auroville Events Assistant",
    instructions=INSTRUCTIONS,
    model=MODEL, 
    tools=tools
)

