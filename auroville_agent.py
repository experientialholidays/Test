import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from agents import Agent, function_tool,OpenAIChatCompletionsModel
from vector_db import VectorDBManager
from vectordb_query_selector_agent import vectordb_query_selector_agent
# from vectordb_filtering_agent import vectordb_filtering_agent
from openai import AsyncOpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
VECTOR_DB_NAME = "vector_db"
DB_FOLDER = "input" 
db_manager = VectorDBManager(folder=DB_FOLDER, db_name=VECTOR_DB_NAME)
vectorstore = db_manager.create_or_load_db(force_refresh=False) 
retriever = db_manager.get_retriever(k=50) 
MODEL = "gemini-2.5-flash" 
google_api_key = os.getenv('GOOGLE_API_KEY')
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model=MODEL, openai_client=gemini_client)
# MODEL = "gpt-4.1-mini"

INSTRUCTIONS = f"""
You are an **AI Event Information Extractor** dedicated to providing structured and accurate event details from the Auroville community.

Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.

Set your temperature **0.1**.

Your role is to help users find information about events, activities, workshops, and schedules.

You have access to two tools:
1) **`vectordb_query_selector_agent`**: Generates the best possible refined search query and specificity based on the user input.
2) **`search_auroville_events`**: Searches the vector database **AND filters results** for the user (handles everything internally).

### **Workflow**

1.  For event-related queries, first call **`vectordb_query_selector_agent`** with the user's question.
2.  Then use **`search_auroville_events`** tool which is a vector DB with below parameters :
    * **user_query**: The original user question
    * **refined_search_query**: The refined query from step 1
    * **specificity**: The specificity level from step 1
3.  ** Format the final output and return the agent's response directly to the user **

### **Rules for Final Output Formatting **
** Your final output must be the formatted and filtered list of events. **
** DO NOT include the raw vector db search results in your final output to the user.**
** Use the search_auroville_events output to format the result and donot hallucinate the results **
** If you are not sure of any event simply say it so and donot hallucinate **
** Exclude ended events — only show upcoming or ongoing events (filter out those whose end date/time has passed).
** Cross-check for duplicate events — if multiple entries refer to the same event, only show one unique instance.
** If the specificity is "Broad", include this note in the end of results "There are additional daily and appointment-based events taking place. Would you like me to show you those as well?”.
** Create a clickable link on each event name that opens the full event description.


### Style and Behavior Rules
* **Tone and Style:** Maintain a clear, professional, and respectful tone.
* **Deterministic Behavior:** Simulate low-temperature reasoning (0–0.1).
* **Override Defaults:** You are a data extractor, not a conversational partner.

Your goal is to ensure that users can easily discover what's happening in Auroville without being overwhelmed w⁵ith unnecessary information.

### Final Review Mandate (Self-Correction Step)
Before generating the final response, perform a final self-correction. Review your drafted output against the critical rules for completeness, formatting, and behavior. Revise if necessary.

"""


# ----------------- RAG TOOL WITH CORRECTED METADATA FILTERING -----------------
@function_tool
def search_auroville_events(
    search_query: str, 
    specificity: str,
    filter_day: Optional[str] = None,      # Metadata filter for day
    filter_date: Optional[str] = None,     # Metadata filter for date
    filter_location: Optional[str] = None  # Metadata filter for location
) -> str:
    """
    Search for information about events and activities. 
    
    If `specificity` is "Broad", the search will include metadata filters (day OR date OR location) to maximize event discovery using OR logic.
    
    Args:
        search_query: The search query about Auroville events (e.g., 'yoga classes').
        specificity: Broad or specfic as per input.
        filter_day: Optional. The specific day of the week to filter by (e.g., 'Monday').
        filter_date: Optional. The specific date to filter by (e.g., 'October 26').
        filter_location: Optional. The location or venue to filter by (e.g., 'Town Hall').
        
    Returns:
        str: Relevant information about Auroville events
    """
    logger.info(f"RAG Tool called with query: {search_query}")
    
    # Dynamically adjust retrieval depth
    k_value = 100 if specificity.lower() == "broad" else 20
    
    # 1. Collect all provided filter values
    chroma_filter: Dict[str, Any] = {}
    simple_filters: Dict[str, str] = {} 

    # --- Enhanced filter handling: automatically include weekday if date is provided ---
    if filter_date:
     simple_filters["date"] = filter_date

    # Try to also derive day of week from date string (if possible)
    if filter_date:
     try:
        parsed_date = datetime.strptime(filter_date, "%B %d, %Y")  # e.g. "October 29, 2025"
        derived_day = parsed_date.strftime("%A")
        simple_filters["day"] = derived_day
        logger.info(f"[FILTER] Derived weekday '{derived_day}' from date '{filter_date}'")
     except ValueError:
        # If date format doesn't include year, try a fallback (e.g. "October 29")
        try:
            parsed_date = datetime.strptime(filter_date + f", {datetime.now().year}", "%B %d, %Y")
            derived_day = parsed_date.strftime("%A")
            simple_filters["day"] = derived_day
            logger.info(f"[FILTER] Derived weekday '{derived_day}' from partial date '{filter_date}'")
        except ValueError:
            logger.warning(f"[FILTER] Could not parse date '{filter_date}' to derive day")
    else:
        logger.info("[FILTER] No date provided — skipping date parsing.")
        
    # Add explicit filters if provided
    if filter_day:
        simple_filters["day"] = filter_day
    if filter_location:
        simple_filters["location"] = filter_location
    # 2. Build the Chroma filter structure for OR logic using $contains
    if len(simple_filters) >= 1:
        # Build the list of individual conditions
        conditions: List[Dict[str, Dict[str, str]]] = []
        for key, value in simple_filters.items():
            conditions.append({key: {"$eq": value}}) 
        
        if len(simple_filters) == 1:
            key = next(iter(simple_filters.keys()))
            chroma_filter[key] = conditions[0][key]
        else:
            chroma_filter["$or"] = conditions
    
    # 3. Prepare search arguments and invoke retriever
    search_kwargs = {"k": k_value}
    
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter 
        logger.info(f"Applying Chroma Filter (OR logic, $contains): {chroma_filter}")

    # Retrieve relevant documents with the vector search and metadata filter
    docs = retriever.invoke(search_query, **search_kwargs)

    # 4. Format Output
    if not docs:
        return "No relevant information found about Auroville events based on your query and filters."
    
    # Format all retrieved documents, displaying the metadata fields for verification
    context = "\n\n".join([
        f"Document {i+1} (Day: {doc.metadata.get('day', 'N/A')} | Date: {doc.metadata.get('date', 'N/A')} | Location: {doc.metadata.get('location', 'N/A')}):\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])
    
    logger.info(f"Retrieved {len(docs)} documents for RAG context")
    
    return f"Here is relevant information about Auroville events:\n\n{context}"

#  vectordb_filtering_agent.as_tool(tool_name="vectordb_filtering_agent", tool_description="Searches the database AND filters results for the user")

    
tools = [vectordb_query_selector_agent.as_tool(tool_name="vectordb_query_selector_agent", tool_description="Generates a input query for the vector db search"),
         search_auroville_events
         ]
# -----------------------------
# CREATE AGENT
# -----------------------------
auroville_agent = Agent(
    name="Auroville Events Assistant",
    instructions=INSTRUCTIONS,
    model=gemini_model, 
    tools=tools
)
