import os
import logging
import urllib.parse
from datetime import datetime
from typing import Optional, Dict, Any, List

from agents import Agent, function_tool, OpenAIChatCompletionsModel
from vector_db import VectorDBManager
from vectordb_query_selector_agent import vectordb_query_selector_agent 
from openai import AsyncOpenAI

# -------------------------------------------------------------------------
# 1. Setup & Global Cache
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- IN-MEMORY CACHE FOR EVENTS ---
# Keys: URLEncoded Titles (e.g., "Clay%20Workshop")
# Values: The full Document object (metadata + page_content)
EVENT_DATA_STORE = {} 

VECTOR_DB_NAME = "vector_db"
DB_FOLDER = "input"
MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv('GOOGLE_API_KEY')

db_manager = VectorDBManager(folder=DB_FOLDER, db_name=VECTOR_DB_NAME)
vectorstore = db_manager.create_or_load_db(force_refresh=False)
retriever = db_manager.get_retriever(k=50)

gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model=MODEL, openai_client=gemini_client)

# -------------------------------------------------------------------------
# 2. Formatting Helpers
# -------------------------------------------------------------------------

def format_event_card(doc_metadata: Dict, doc_content: str) -> str:
    """
    Strict Card Format for Full Details
    """
    title = doc_metadata.get('title', 'Event')
    date_str = doc_metadata.get('date', 'Upcoming')
    day_str = doc_metadata.get('day', '')
    time_str = doc_metadata.get('time', '')
    location = doc_metadata.get('location', 'Unknown Location')
    contribution = doc_metadata.get('contribution', 'Check details')
    contact_info = doc_metadata.get('contact', '')
    description = doc_content.strip()
    poster_url = doc_metadata.get('poster_url', None)
    phone_number = doc_metadata.get('phone', '')

    wa_section = ""
    if phone_number:
        clean_phone = ''.join(filter(str.isdigit, str(phone_number)))
        if clean_phone:
            msg = f"Hi, I came across your event '{title}' scheduled on {date_str}. Info?"
            encoded_msg = urllib.parse.quote(msg)
            wa_url = f"https://wa.me/{clean_phone}?text={encoded_msg}"
            wa_section = f"\n[**Click to Chat on WhatsApp**]({wa_url})"

    card = f"""
**Event Name:** {title}
**When:** {day_str}, {date_str} @ {time_str}
**Where:** {location}
**Contribution:** {contribution}
**Contact:** {contact_info}{wa_section}

**Description:**
{description}
"""
    if poster_url:
        card += f"\n\n![Event Poster]({poster_url})"
    
    return card.strip()

def format_summary_line(doc_metadata: Dict) -> str:
    """
    Formats the summary line with a SPECIAL fetch link.
    """
    title = doc_metadata.get('title', 'Event')
    day = doc_metadata.get('day', '')
    time = doc_metadata.get('time', '')
    loc = doc_metadata.get('location', '')
    
    # Create a safe key for the cache lookup
    safe_key = urllib.parse.quote(title)
    
    # Link format: #FETCH::EncodedTitle
    return f"- [**{title}**](#FETCH::{safe_key}) | {day} {time} @ {loc}"

# -------------------------------------------------------------------------
# 3. Optimized Tool (Populates Cache)
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
    Searches events, CACHES them in memory, and returns formatted text.
    """
    # ... (Search Logic same as before) ...
    k_value = 100 if specificity.lower() == "broad" else 15
    chroma_filter = {}
    simple_filters = {}

    if filter_date:
        simple_filters["date"] = filter_date
        try:
            for fmt in ["%B %d, %Y", "%B %d"]:
                try:
                    parse_str = filter_date if "Y" in fmt else f"{filter_date}, {datetime.now().year}"
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
        
        # Handle internal list deduplication
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
            final_output.append("\n\n[**See daily & appointment-based events**](#trigger_broad_search)")

    return "\n".join(final_output)


INSTRUCTIONS = f"""
You are the **Auroville Events Assistant**.
Today is {datetime.now().strftime("%A, %B %d, %Y")}.

1. Use **`vectordb_query_selector_agent`** to refine the query.
2. Use **`search_auroville_events`** to find events.
3. **PASS THROUGH** the exact output from the tool. Do not reformat.
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
