import os
import logging
import urllib.parse
import ast
from datetime import datetime
from typing import Optional, Dict, Any, List

from agents import Agent, function_tool, OpenAIChatCompletionsModel
from vector_db import VectorDBManager, get_event_level, parse_time_for_sort, is_date_specific # NEW IMPORTS
from vectordb_query_selector_agent import vectordb_query_selector_agent 
from openai import AsyncOpenAI
from langchain_core.documents import Document 

# -------------------------------------------------------------------------
# 1. Setup & Global Cache (Lazy Initialization Fix Applied Here)
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVENT_DATA_STORE: Dict[str, Document] = {} 

VECTOR_DB_NAME = "vector_db"
DB_FOLDER = "input"
MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv('GOOGLE_API_KEY')

db_manager = VectorDBManager(folder=DB_FOLDER, db_name=VECTOR_DB_NAME)

# --- CRITICAL FIX: LAZY INITIALIZATION ---
retriever = None 

def initialize_retriever(vectorstore):
    """Initializes the global retriever instance after vectorstore creation."""
    global retriever
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
        logger.info("Auroville agent retriever successfully initialized.")
    else:
        logger.error("Failed to initialize retriever: Vectorstore is None.")

gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model=MODEL, openai_client=gemini_client)

# -------------------------------------------------------------------------
# 2. Formatting Helpers 
# -------------------------------------------------------------------------

def format_event_card(doc_metadata: Dict, doc_content: str) -> str:
    """
    Strict Card Format for Full Details, suppressing empty fields.
    """
    title = doc_metadata.get('title', 'Event').strip()
    date_str = doc_metadata.get('date', '').strip()
    day_str = doc_metadata.get('day', '').strip()
    time_str = doc_metadata.get('time', '').strip()
    location = doc_metadata.get('location', '').strip()
    contribution = doc_metadata.get('contribution', '').strip()
    contact_info = doc_metadata.get('contact', '').strip()
    description = doc_content.strip()
    poster_url = doc_metadata.get('poster_url', None)
    phone_number = doc_metadata.get('phone', '').strip()

    output_lines = []
    
    output_lines.append(f"**Event Name:** {title}")

    when_parts = []
    if day_str:
        when_parts.append(day_str)
    if date_str:
        when_parts.append(date_str)
    if time_str:
        when_parts.append(f"@ {time_str}")
        
    if when_parts:
        output_lines.append(f"**When:** {' '.join(when_parts)}")
    
    if location:
        output_lines.append(f"**Where:** {location}")

    if contribution:
        output_lines.append(f"**Contribution:** {contribution}")
        
    wa_section = ""
    clean_phone = ''
    if phone_number:
        clean_phone = ''.join(filter(str.isdigit, str(phone_number)))
        if clean_phone:
            msg = f"Hi, I came across your event '{title}' scheduled on {date_str}. Info?"
            encoded_msg = urllib.parse.quote(msg)
            wa_url = f"https://wa.me/{clean_phone}?text={encoded_msg}"
            wa_section = f"\n[**Click to Chat on WhatsApp**]({wa_url})"
            
    if contact_info or clean_phone:
        contact_line = f"**Contact:** {contact_info}" if contact_info else "**Contact:**"
        output_lines.append(f"{contact_line}{wa_section}")

    output_lines.append("\n**Description:**")
    output_lines.append(description)

    if poster_url:
        output_lines.append(f"\n\n![Event Poster]({poster_url})")
    
    return "\n".join(filter(None, output_lines)).strip()


def format_summary_line(doc_metadata: Dict) -> str:
    """
    Formats the summary line with a SPECIAL fetch link, including key info.
    """
    title = doc_metadata.get('title', 'Event').strip()
    day = doc_metadata.get('day', '').strip()
    time_str = doc_metadata.get('time', '').strip()
    loc = doc_metadata.get('location', '').strip()
    contribution = doc_metadata.get('contribution', '').strip() 
    phone_number = doc_metadata.get('phone', '').strip() 
    
    safe_key = urllib.parse.quote(title)
    
    summary_parts = [f"- [**{title}**](#FETCH::{safe_key})"]
    
    details = []
    
    if day:
        details.append(day)
    if time_str:
        details.append(time_str)
    if loc:
        details.append(f"@{loc}")
    
    if contribution:
        details.append(f"| Contrib: {contribution}")
        
    clean_phone = ''.join(filter(str.isdigit, str(phone_number)))
    if clean_phone:
        details.append(f"| Ph: {clean_phone}")
    
    if details:
        summary_parts.append("|")
        summary_parts.append(" ".join(details))
        
    return " ".join(summary_parts)

# -------------------------------------------------------------------------
# 3. Optimized Tool (Full Body with Sorting and Filtering)
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
    ... (Docstring truncated for brevity) ...
    """
    global retriever
    if retriever is None:
        logger.error("Retriever is not initialized. Cannot perform search.")
        return "The event database is still initializing. Please wait a moment and try again."

    k_value = 100 if specificity.lower() == "broad" else 12
    chroma_filter = {}
    simple_filters = {}

    # ... (Filter logic unchanged) ...
    if filter_date:
        simple_filters["date"] = filter_date
        # Current date is Saturday, November 15, 2025
        try:
            # Add formats for robust parsing, including numeric ones like 30.11.25
            for fmt in ["%B %d, %Y", "%B %d", "%d %B", "%d %b", "%d.%m.%y", "%d.%m.%Y"]:
                try:
                    current_year = datetime.now().year
                    # Append current year if not present in the format string
                    parse_str = filter_date
                    if "%Y" not in fmt and "%y" not in fmt:
                         parse_str = f"{filter_date}, {current_year}"

                    dt = datetime.strptime(parse_str.strip(), fmt.strip())
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

    # --- FILTERING (EXCLUDE ENDED EVENTS) ---
    current_datetime = datetime.now()
    current_date = current_datetime.date()
    current_time = current_datetime.time()
    
    filtered_docs = []
    
    for doc in docs:
        date_str = doc.metadata.get('date', '').strip()
        time_str = doc.metadata.get('time', '').strip()
        is_specific = is_date_specific(date_str, doc.metadata.get('day', ''))
        
        if is_specific:
            try:
                # Attempt robust date parsing for comparison
                event_datetime = None
                
                # Use robust formats again (same as above)
                for fmt in ["%B %d, %Y", "%B %d", "%d %B", "%d %b", "%d.%m.%y", "%d.%m.%Y"]:
                    try:
                        parse_str = date_str
                        if "%Y" not in fmt and "%y" not in fmt:
                             parse_str = f"{date_str}, {current_date.year}"
                             
                        event_date = datetime.strptime(parse_str.strip(), fmt.strip()).date()
                        event_time = parse_time_for_sort(time_str)
                        event_datetime = datetime.combine(event_date, event_time)
                        break
                    except ValueError:
                        continue
                        
                if event_datetime:
                     # Filter: If the event date is in the past, or if it's today and the time is past 23:59 (the default sort time)
                    if event_datetime.date() < current_date:
                        continue 
                    if event_datetime.date() == current_date and event_datetime.time() < current_time:
                        continue 

                    filtered_docs.append(doc)
                else:
                    # If date parsing fails but it looks like a specific date, keep it for safety.
                    filtered_docs.append(doc)
            except:
                filtered_docs.append(doc) # Fallback
        else:
            # Keep weekday/daily/appointment events (they are generally ongoing)
            filtered_docs.append(doc)

    if not filtered_docs:
        return "I couldn't find any upcoming or ongoing events matching those criteria."

    docs = filtered_docs
    # --- END FILTERING ---

    # --- GROUPING AND SORTING ---
    grouped_events = {
        "Date-specific": [],
        "Weekday-based": [],
        "Appointment/Daily-based": [],
    }

    # 1. Group documents and assign sort keys
    for doc in docs:
        level = get_event_level(doc.metadata)
        doc.metadata['_level'] = level
        doc.metadata['_sort_time'] = parse_time_for_sort(doc.metadata.get('time', ''))
        grouped_events[level].append(doc)

    # 2. Sort documents within each group by time
    for level, doc_list in grouped_events.items():
        doc_list.sort(key=lambda d: d.metadata['_sort_time'])

    # 3. Create the final ordered list
    final_ordered_docs = []
    order = ["Date-specific", "Weekday-based", "Appointment/Daily-based"]
    
    for level in order:
        final_ordered_docs.extend(grouped_events[level])

    # Remove duplicates
    EVENT_DATA_STORE.clear()
    seen_keys = set()
    unique_docs = []
    
    for doc in final_ordered_docs:
        title = doc.metadata.get('title', 'Unknown')
        safe_key = urllib.parse.quote(title)
        
        EVENT_DATA_STORE[safe_key] = doc
        
        dedup_key = (title, doc.metadata.get('date'), doc.metadata.get('day'))
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            unique_docs.append(doc)
    
    # --- OUTPUT FORMATTING (Uses grouping headers) ---
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
        
        current_level = None
        for doc in unique_docs:
            level = get_event_level(doc.metadata) 
            if level != current_level:
                # Add a bold header for each new group
                final_output.append(f"\n## 📅 **{level} Events**")
                current_level = level
            final_output.append(format_summary_line(doc.metadata))
            
        if specificity.lower() == "broad":
            final_output.append("\n\n[**See daily & appointment-based events**](#TRIGGER_SEARCH::daily and appointment-based events::Broad)")

    return "\n".join(final_output)

# -------------------------------------------------------------------------
# 4. Agent Instructions and Definition (Final Formatting Step Included)
# -------------------------------------------------------------------------

INSTRUCTIONS = f"""
You are an **AI Event Information Extractor** dedicated to providing structured and accurate event details from the Auroville community.

Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.

Set your temperature **0.1**.

Your role is to help users find information about events, activities, workshops, and schedules.

You have access to two tools:
1) **`vectordb_query_selector_agent`**: Generates the best possible refined search query and specificity based on the user input.
2) **`search_auroville_events`**: Searches the vector database and filters results.

### **Event Rules Applied by Search Tool**

The `search_auroville_events` tool automatically applies these rules:
* **Filtering:** Only shows **upcoming or ongoing** events; events whose date/time has passed are excluded.
* **Grouping & Sorting:** Events are grouped by type and then sorted chronologically by time:
    1. **Date-specific** events (e.g., Nov 14th)
    2. **Weekday-based** events (e.g., Every Monday)
    3. **Appointment/Daily-based** events
* **Do not hallucinate any event details. If a specific event is not found, state that you are not sure.**

### **Workflow**

1.  **SPECIAL RULE:** If the user's question is exactly "daily and appointment-based events", you **must skip** calling `vectordb_query_selector_agent` and directly call `search_auroville_events` with the following parameters:
    * search_query: "daily and appointment-based events"
    * specificity: "Broad"
    * filter_day, filter_date, filter_location: null

2.  For all other event-related queries, first call **`vectordb_query_selector_agent`** with the user's question.
3.  Then use **`search_auroville_events`** tool with the outputs from step 1 or step 2.

4. **FINAL STEP (CRITICAL):** Once you receive the output from the `search_auroville_events` tool, you **MUST** process it. Your final response to the user must be the **exact content** of the tool output, but with these minor cleanups:
    * **Review for Grammatical Errors:** Fix any minor grammatical or spelling mistakes.
    * **Ensure Basic Formatting:** Ensure the markdown formatting (like `**bolding**`, headers, and lists) is clean and consistent before presenting it to the user. Delete repitation of things.
    * **Do not add any preamble or conversational text; just output the cleaned-up event list.**
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
