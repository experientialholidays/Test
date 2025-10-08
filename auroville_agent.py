import os
from datetime import datetime
from agents import Agent, function_tool
from vector_db import VectorDBManager
from vectordb_query_selector_agent import vectordb_query_selector_agent
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
When users ask about events or activities, use the `vectordb_query_selector_agent` tool to generate the input query for vector db and then use
`search_auroville_events` tool to find relevant information.

Use the following rules while filtering the data :-
1) Use the 'vectordb_query_selector_agent' tool first to generate the input query required for searching in vector db.
2) Then use the tool  'search_auroville_events' to find the top relevant matchies from vector db.
3) Monday—Saturday means all the week days from Monday to Saturday. So if user is asking for say thursday event, then Monday-Saturday should also be considered
4) Appointment event means that they are on appointment basis and these events should also be considered for any date and weekday if user is interested in this type of event
5) Provide clear, consise, and helpful responses based on the information you retrieve. 
6) If you don't find relevant information, politely let the user know 
7) Until user asks for description, donot provide the detailed description of events

You **must always** use the tool `vectordb_query_selector_agent` to first transform any user query about events, dates, or schedules into a **refined search query**. 
Then you **must** use the `search_auroville_events` tool with that refined query to fetch results.

Example reasoning chain:
1. User asks: "What's happening today?"
2. You call `vectordb_query_selector_agent` with "today's events in Auroville"
3. You take its response (refined query text)
4. You call `search_auroville_events` using that query
5. You summarize results clearly.

If the user asks anything unrelated to events, respond conversationally without tools.
"""

@function_tool
def search_auroville_events(query: str) -> str:
    """
    Search for information about Auroville events and activities. 
    Use this tool whenever the user asks about events, activities, schedules, or anything related to Auroville.
    
    Args:
        query: The search query about Auroville events (e.g., 'dance events in October', 'music workshops', 'yoga classes')
        
    Returns:
        str: Relevant information about Auroville events
    """
    print(f"RAG Tool called with query: {query}")
    
    # Retrieve relevant documents (uses k=50 from retriever config)
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "No relevant information found about Auroville events."
    
    # Format all retrieved documents
    context = "\n\n".join([
        f"Document {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])
    
    print(f"📚 Retrieved {len(docs)} documents for RAG context")
    
    return f"Here is relevant information about Auroville events:\n\n{context}"


tools = [search_auroville_events,
         vectordb_query_selector_agent.as_tool(tool_name="vectordb_query_selector_agent", tool_description="Generates a input query for the vector db search")]
# -----------------------------
# CREATE AGENT
# -----------------------------
auroville_agent = Agent(
    name="Auroville Events Assistant",
    instructions=INSTRUCTIONS,
    model=MODEL, 
    tools=tools
)

