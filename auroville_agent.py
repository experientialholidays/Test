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

**Usage Rules:**
1. Always start by using `vectordb_query_selector_agent` to refine any user query about events, dates, or schedules.
2. Use `search_auroville_events` with that refined query and specificity to fetch relevant results.
3. Then call `vectordb_filtering_agent` to process and present the filtered, user-friendly output.
4. If the user asks about something unrelated to events or Auroville, reply conversationally without using tools.
5. If no relevant events are found, politely inform the user.

Example workflow:
1. User asks: "What's happening today?"
2. Call `vectordb_query_selector_agent` with "today's events in Auroville"
3. Use its response with `search_auroville_events`
4. Pass results to `vectordb_filtering_agent`
5. Return the filtered summary to the user.
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
         vectordb_filtering_agent.as_tool(tool_name="vectordb_filtering_agent", tool_description="Formats and filter out the vector db results for user")]
# -----------------------------
# CREATE AGENT
# -----------------------------
auroville_agent = Agent(
    name="Auroville Events Assistant",
    instructions=INSTRUCTIONS,
    model=MODEL, 
    tools=tools
)

