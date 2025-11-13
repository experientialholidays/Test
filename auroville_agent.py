
import os
import logging
import urllib.parse
import ast
from datetime import datetime
from typing import Optional, Dict, Any, List

from agents import Agent, function_tool, OpenAIChatCompletionsModel
from vector_db import VectorDBManager
from vectordb_query_selector_agent import vectordb_query_selector_agent 
from openai import AsyncOpenAI
from langchain_core.documents import Document # Added for type hinting

# -------------------------------------------------------------------------
# 1. Setup & Global Cache
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- IN-MEMORY CACHE FOR EVENTS ---
EVENT_DATA_STORE: Dict[str, Document] = {} 

VECTOR_DB_NAME = "vector_db"
DB_FOLDER = "input"
MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv('GOOGLE_API_KEY')

db_manager = VectorDBManager(folder=DB_FOLDER, db_name=VECTOR_DB_NAME)
# NOTE: The actual vectorstore loading (db_manager.create_or_load_db) is in app.py.
# We just initialize the retriever here.
retriever = db_manager.get_retriever(k=50) 

gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model=MODEL, openai_client=gemini_client)

# -------------------------------------------------------------------------
# 2. Formatting Helpers (Updated)
# -------------------------------------------------------------------------

def format_event_card(doc_metadata: Dict, doc_content: str) -> str:
    """
    Strict Card Format for Full Details, suppressing empty fields.
    """
    # 1. Get all data, stripping whitespace and using safe defaults
    title = doc_metadata.get('title', 'Event').strip()
    date_str = doc_metadata.get('date', 'Upcoming').strip()
    day_str = doc_metadata.get('day', '').strip()
    time_str = doc_metadata.get('time', '').strip()
    location = doc_metadata.get('location', 'Unknown Location').strip()
    contribution = doc_metadata.get('contribution', 'Check details').strip()
    contact_info = doc_metadata.get('contact', '').strip()
    description = doc_content.strip()
    poster_url = doc_metadata.get('poster_url', None)
    phone_number = doc_metadata.get('phone', '').strip()

    output_lines = []
    
    # Define common placeholders to ignore
    placeholders = {'n/a', 'unknown location', 'upcoming', 'check details'}

    # 2. Event Name (Always included, uses default 'Event' if missing)
    output_lines.append(f"**Event Name:** {title}")

    # 3. When/Time Section (Conditional)
    when_parts = []
    if day_str and day_str.lower() not in placeholders:
        when_parts.append(day_str)
    if date_str and date_str.lower() not in placeholders:
        when_parts.append(date_str)
    if time_str and time_str.lower() not in placeholders:
        when_parts.append(f"@ {time_str}")
        
    if when_parts:
        output_lines.append(f"**When:** {' '.join(when_parts)}")
    
    # 4. Location (Conditional)
    if location and location.lower() not in placeholders:
        output_lines.append(f"**Where:** {location}")

    # 5. Contribution (Conditional)
    if contribution and contribution.lower() not in placeholders:
        output_lines.append(f"**Contribution:** {contribution}")
        
    # 6. Contact & WhatsApp (Conditional)
    wa_section = ""
    clean_phone = ''
    if phone_number:
        clean_phone = ''.join(filter(str.isdigit, str(phone_number)))
        if clean_phone:
            msg = f"Hi, I came across your event '{title}' scheduled on {date_str}. Info?"
            encoded_msg = urllib.parse.quote(msg)
            wa_url = f"https://wa.me/{clean_phone}?text={encoded_msg}"
            wa_section = f"\n[**Click to Chat on WhatsApp**]({wa_url})"
            
    # Include Contact line only if contact_info exists OR if a WhatsApp link was created
    if contact_info or clean_phone:
        contact_line = f"**Contact:** {contact_info}" if contact_info else "**Contact:**"
        output_lines.append(f"{contact_line}{wa_section}")

    # 7. Description (Always included, even if empty)
    output_lines.append("\n**Description:**")
    output_lines.append(description)

    # 8. Poster URL
    if poster_url and poster_url.lower() != 'none':
        output_lines.append(f"\n\n![Event Poster]({poster_url})")
    
    # Filter out empty strings and join
    return "\n".join(filter(None, output_lines)).strip()


def format_summary_line(doc_metadata: Dict) -> str:
    """
    Formats the summary line with a SPECIAL fetch link, including key info.
    """
    title = doc_metadata.get('title', 'Event').strip()
    day = doc_metadata.get('day', '').strip()
    time = doc_metadata.get('time', '').strip()
    loc = doc_metadata.get('location', '').strip()
    contribution = doc_metadata.get('contribution', '').strip() # NEW
    phone_number = doc_metadata.get('phone', '').strip() # NEW
    
    placeholders = {'n/a', 'unknown location', 'upcoming', 'check details'}

    # Create a safe key for the cache lookup
    safe_key = urllib.parse.quote(title)
    
    # Start with the fetch link
    summary_parts = [f"- [**{title}**](#FETCH::{safe_key})"]
    
    # Add optional parts only if they exist and are not placeholders
    details = []
    
    # 1. Day, Time, Location
    if day and day.lower() not in placeholders:
        details.append(day)
    if time and time.lower() not in placeholders:
        details.append(time)
    if loc and loc.lower() not in placeholders:
        details.append(f"@{loc}")
    
    # 2. Contribution (NEW)
    if contribution and contribution.lower() not in placeholders:
        details.append(f"| Contrib: {contribution}")
        
    # 3. Contact Phone (NEW, only if digits exist)
    clean_phone = ''.join(filter(str.isdigit, str(phone_number)))
    if clean_phone:
        details.append(f"| Ph: {clean_phone}")
    
    if details:
        summary_parts.append("|")
        summary_parts.append(" ".join(details))
        
    return " ".join(summary_parts)

# -------------------------------------------------------------------------
# 3. Optimized Tool (Full Body)
# -------------------------------------------------------------------------
@function_tool
def search_auroville_events(
    search_query: str, 
    specificity: str,
    filter_day: Optional[str] = None,
    filter_date: Optional[str] = None,
    filter_location: Optional[str] = None
) -> str:
    """
    Search for information about events and activities. 
    
    If `specificity` is "Broad", the search will include metadata filters (day OR date OR location) to maximize event discovery using OR logic.
    
    Args:
        search_query: The search query about Auroville events (e.g., 'yoga classes').
        specificity: Broad or specfic as per input.
        filter_day: Optional. The specific day of the week to filter by (e.g., 'Monday').
        filter_date: Optional. The specific date to filter by (e.g., 'November 13, 2025').
        filter_location: Optional. The location or venue to filter by (e.g., 'Town Hall').
        
    Returns:
        str: Relevant information about Auroville events.
        
    Searches events, CACHES them in memory, and returns formatted text.
    """
    k_value = 100 if specificity.lower() == "broad" else 12
    chroma_filter = {}
    simple_filters = {}

    # Current date is Thursday, November 13, 2025
    
    if filter_date:
        simple_filters["date"] = filter_date
        try:
            for fmt in ["%B %d, %Y", "%B %d"]:
                try:
                    # Use a robust way to parse the year for the current context
                    current_year = datetime.now().year
                    parse_str = filter_date if "Y" in fmt else f"{filter_date}, {current_year}"
                    dt = datetime.strptime(parse_str, fmt)
                    simple_filters["day"] = dt.strftime("%A")
                    break
                except ValueError: continue
        except: pass

    if filter_day: simple_filters["day"] = filter_day
    if filter_location: simple_filters["location"] = filter_location

    if len(simple_filters) == 1:
        k, v = list(simple_filters.items())[0]
        chroma_filter[k] = {"$eq": v}
    elif len(simple_filters) > 1:
        chroma_filter["$or"] = [{k: {"$eq": v}} for k, v in simple_filters.items()]

    search_kwargs = {"k": k_value}
    if chroma_filter: search_kwargs["filter"] = chroma_filter

    # EXECUTE SEARCH
    docs = retriever.invoke(search_query, **search_kwargs)

    if not docs:
        return "I couldn't find any upcoming events matching those criteria."

    # --- POPULATE CACHE HERE ---
    global EVENT_DATA_STORE
    
    # Deduplicate
    seen_keys = set()
    unique_docs = []
    
    for doc in docs:
        # Create cache key
        title = doc.metadata.get('title', 'Unknown')
        safe_key = urllib.parse.quote(title)
        
        # Store in Global Cache (Overwrites old versions, which is good)
        EVENT_DATA_STORE[safe_key] = doc
        
        # Handle internal list deduplication (e.g., same title, same date)
        dedup_key = (title, doc.metadata.get('date'))
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            unique_docs.append(doc)

    count = len(unique_docs)
    final_output = []
    
    is_exact_match = (count > 0 and search_query.lower() in unique_docs[0].metadata.get('title', '').lower())

    if count < 5 or specificity.lower() == "specific" or is_exact_match:
        final_output.append(f"Found {count} event(s):\n")
        for doc in unique_docs:
            final_output.append(format_event_card(doc.metadata, doc.page_content))
            final_output.append("\n" + "="*30 + "\n")
    else:
        final_output.append(f"Found {count} events. Click a name to see details:\n")
        for doc in unique_docs:
            final_output.append(format_summary_line(doc.metadata))
            
        if specificity.lower() == "broad":
            # Special link for broad search
            final_output.append("\n\n[**See daily & appointment-based events**](#TRIGGER_SEARCH::daily and appointment-based events::Broad)")

    return "\n".join(final_output)

# -------------------------------------------------------------------------
# 4. Agent Instructions and Definition
# -------------------------------------------------------------------------

INSTRUCTIONS = f"""
You are an **AI Event Information Extractor** dedicated to providing structured and accurate event details from the Auroville community.

Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.

Set your temperature **0.1**.

Your role is to help users find information about events, activities, workshops, and schedules.

You have access to two tools:
1) **`vectordb_query_selector_agent`**: Generates the best possible refined search query and specificity based on the user input.
2) **`search_auroville_events`**: Searches the vector database **AND filters results** for the user (handles everything internally).

### **Workflow**

1.  **SPECIAL RULE:** If the user's question is exactly "daily and appointment-based events", you **must skip** calling `vectordb_query_selector_agent` and directly call `search_auroville_events` with the following parameters:
    * search_query: "daily and appointment-based events"
    * specificity: "Broad"
    * filter_day, filter_date, filter_location: null

2.  For all other event-related queries, first call **`vectordb_query_selector_agent`** with the user's question.
3.  Then use **`search_auroville_events`** tool with the outputs from step 1 or step 2.
4. **PASS THROUGH** the exact output from the tool. Do not reformat.
"""

tools = [
    vectordb_query_selector_agent.as_tool(tool_name="vectordb_query_selector_agent", tool_description="Refines query."),
    search_auroville_events
]

auroville_agent = Agent(
    name="Auroville Events Assistant",
    instructions=INSTRUCTIONS,
    model=gemini_model, 
    tools=tools
)
