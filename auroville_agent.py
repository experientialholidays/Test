import os
import logging
import urllib.parse
from datetime import datetime, time, date
from typing import Optional, Dict, Any, List
import re

from agents import Agent, function_tool, OpenAIChatCompletionsModel
from vector_db import VectorDBManager
from vectordb_query_selector_agent import vectordb_query_selector_agent
from openai import AsyncOpenAI
from langchain_core.documents import Document

# -------------------------------------------------------------------------
# Date/Time Helper Functions
# -------------------------------------------------------------------------

def is_date_specific(date_str, day_str):
    """Classifies an event as date-specific."""
    return bool(date_str and str(date_str).strip().lower() not in ('', 'n/a', 'upcoming', 'none'))

# NEW HELPER FUNCTION FOR DATE STRING PARSING
def _parse_date_string(date_str: str, year: int) -> Optional[date]:
    """Robustly parses a date string into a datetime.date object."""
    if not date_str:
        return None
    DATE_FORMATS = ["%B %d, %Y", "%B %d", "%d %B", "%d %b", "%d.%m.%y", "%d.%m.%Y", "%Y-%m-%d"]
    
    for fmt in DATE_FORMATS:
        try:
            p = date_str
            if "%Y" not in fmt and "%y" not in fmt:
                # If year is missing, assume the provided year
                p = f"{date_str.strip()}, {year}"
            
            # Use 'datetime' from imported 'datetime'
            return datetime.strptime(p.strip(), fmt.strip()).date()
        except:
            continue
    return None

# -------------------------------------------------------------------------
# Robust Time Parser for Sorting
# -------------------------------------------------------------------------

def parse_time_for_sort(raw: str) -> time:
    if not raw:
        return time(23, 59, 59)

    s = str(raw).replace("—", "-").replace("–", "-").replace("-", "-").strip().upper()

    if re.search(r'\bANYTIME\b|\bOPEN\b|\bALL DAY\b', s):
        return time(23, 59, 59)

    token_pattern = re.compile(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?')
    tokens = list(token_pattern.finditer(s))

    if not tokens:
        trailing = re.search(r'(\d{1,2})(?::(\d{2}))?[-\s]*\d{1,2}(?::\d{2})?\s*(AM|PM)', s)
        if trailing:
            h = int(trailing.group(1))
            m = int(trailing.group(2)) if trailing.group(2) else 0
            mer = trailing.group(3)
            if mer == "PM" and h != 12:
                h += 12
            if mer == "AM" and h == 12:
                h = 0
            return time(h, m)
        return time(23, 59, 59)

    chosen = None
    mer = None
    for t in tokens:
        if t.group(3):
            chosen = t
            mer = t.group(3)
            break

    if not chosen:
        chosen = tokens[0]
        after = s[chosen.end(): chosen.end() + 10]
        m_after = re.search(r'\b(AM|PM)\b', after)
        mer = m_after.group(1) if m_after else None

    hour = int(chosen.group(1))
    minute = int(chosen.group(2)) if chosen.group(2) else 0

    if mer == "PM" and hour != 12:
        hour += 12
    if mer == "AM" and hour == 12:
        hour = 0

    if hour < 0 or hour > 23:
        return time(23, 59, 59)
    if minute < 0 or minute > 59:
        minute = 0

    return time(hour, minute)

# -------------------------------------------------------------------------
# 1. Setup & Global Cache
# -------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map numeric index -> Document (cache shown results)
EVENT_DATA_STORE: Dict[int, Document] = {}

VECTOR_DB_NAME = "vector_db"
DB_FOLDER = "input"
MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "[https://generativelanguage.googleapis.com/v1beta/openai/](https://generativelanguage.googleapis.com/v1beta/openai/)"
google_api_key = os.getenv('GOOGLE_API_KEY')

db_manager = VectorDBManager(folder=DB_FOLDER, db_name=VECTOR_DB_NAME)

retriever = None

def initialize_retriever(vectorstore):
    global retriever
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
    else:
        logger.error("Retriever init failed: Vectorstore is None.")

# Initialize DB & retriever eagerly
vectorstore = db_manager.create_or_load_db()
initialize_retriever(vectorstore)

gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model=MODEL, openai_client=gemini_client)

# -------------------------------------------------------------------------
# 2. Formatting Helpers (MODIFIED format_event_card for clickable poster)
# -------------------------------------------------------------------------

def format_event_card(doc_metadata: Dict, doc_content: str) -> str:
    title = doc_metadata.get('title', 'Event').strip()
    date_str = doc_metadata.get('date', '').strip()
    
    # CLEAN THE DAY STRING: Remove [, ], ', ", and ,
    raw_day = doc_metadata.get('day', '').strip() 
    day_str = re.sub(r"[\[\]'\",]", " ", raw_day)
    day_str = re.sub(r"\s+", " ", day_str).strip()

    time_str = doc_metadata.get('time', '').strip()
    location = doc_metadata.get('location', '').strip()
    contribution = doc_metadata.get('contribution', '').strip()
    contact_info = doc_metadata.get('contact', '').strip()
    poster_url = doc_metadata.get('poster_url')
    phone_number = doc_metadata.get('phone', '').strip()
    category = doc_metadata.get('category', '').strip()

    description_meta = doc_metadata.get('description', '').strip()
    email_str = doc_metadata.get('email', '').strip()
    audience_str = doc_metadata.get('audience', '').strip()

    out = []
    out.append(f"**Event Name:** {title}")
    if category:
        out.append(f"**Category:** {category}")

    when = []
    if day_str: when.append(day_str)
    if date_str: when.append(date_str)
    if time_str: when.append(f"@ {time_str}")
    if when:
        out.append(f"**When:** {' '.join(when)}")

    if location:
        out.append(f"**Where:** {location}")
    if contribution:
        out.append(f"**Contribution:** {contribution}")
        
    if audience_str:
        out.append(f"**Target Audience/Prerequisites:** {audience_str}")

    contact_parts = []
    if contact_info: contact_parts.append(contact_info)
    if email_str: contact_parts.append(f"Email: {email_str}")

    clean_phone = ''.join(filter(str.isdigit, phone_number))
    wa_link = ""
    if clean_phone:
        msg = f"Hi, I came across your event '{title}' scheduled on {date_str}. Info?"
        wa = urllib.parse.quote(msg)
        wa_link = f"[**Click to Chat on WhatsApp**](https://wa.me/{clean_phone}?text={wa})"
        contact_parts.append(wa_link)

    if contact_parts:
        display_contact = " | ".join([p for p in contact_parts if p != wa_link])
        if wa_link and wa_link in contact_parts:
             display_contact += f"\n{wa_link}"
        out.append(f"**Contact:** {display_contact}")

    out.append("\n**Description:**")
    out.append(description_meta if description_meta else "No detailed description provided.")

    # --- MODIFICATION: Wrap Markdown image tag in HTML anchor tag for clickability ---
    if poster_url:
        out.append(f"\n\n<a href='{poster_url}' target='_blank'>![Event Poster]({poster_url})</a>")
    # --------------------------------------------------------------------------------

    return "\n".join(out)

def format_summary_numbered(index: int, meta: Dict) -> str:
    title = meta.get('title', '').strip()
    date_str = meta.get('date', '').strip()
    
    raw_day = meta.get('day', '').strip()
    day = re.sub(r"[\[\]'\",]", " ", raw_day)
    day = re.sub(r"\s+", " ", day).strip()

    time_str = meta.get('time', '').strip()
    loc = meta.get('location', '').strip()
    contrib = meta.get('contribution', '').strip()
    phone = meta.get('phone', '').strip()
    audience = meta.get('audience', '').strip()

    parts = []
    if date_str: parts.append(date_str) 
    elif day: parts.append(day) 
    
    if time_str: parts.append(time_str)
    if loc: parts.append(f"@{loc}")
    if contrib: parts.append(f"| Contrib: {contrib}")
    if phone:
        parts.append(f"| Ph:{''.join(filter(str.isdigit, phone))}")
    if audience:
        parts.append(f"| Audience: {audience}")

    line = " ".join(parts)
    return (
        f"{index}. **{title}** — {line}\n"
        f"   👉 <a href='#DETAILS::{index}'>View details **({index})**</a>"
    )

# -------------------------------------------------------------------------
# 3. Tools 
# -------------------------------------------------------------------------

def get_daily_events_core(start_number: int) -> str:
    global EVENT_DATA_STORE, vectorstore
    try:
        raw = vectorstore.get(
            where={"category": "Daily Events"},
            include=["documents", "metadatas"]
        )
    except Exception as e:
        return f"Error fetching daily events: {e}"

    if not raw or not raw.get("ids"):
        return "No Daily Events found."

    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    for d in docs:
        d.metadata["_sort_time"] = parse_time_for_sort(d.metadata.get("time", ""))

    docs.sort(key=lambda d: d.metadata["_sort_time"])

    out = ["\n## 🌞 Daily Events"]
    idx = start_number

    for d in docs:
        idx += 1
        EVENT_DATA_STORE[idx] = d
        out.append(format_summary_numbered(idx, d.metadata))

    return "\n".join(out)

@function_tool
def get_daily_events(start_number: int) -> str:
    """Returns ALL Daily Events using metadata filtering only."""
    return get_daily_events_core(start_number)

@function_tool
def search_auroville_events(
    search_query: str,
    specificity: str,
    filter_day: Optional[str] = None,
    filter_date: Optional[str] = None,
    filter_location: Optional[str] = None
) -> str:

    global retriever, EVENT_DATA_STORE

    if retriever is None:
        return "The event database is still initializing. Please try again."

    k_value = 100 if specificity.lower() == "broad" else 12
    chroma_filter = {}
    simple_filters = {}
    
    now = datetime.now()
    today = now.date()
    now_time = now.time()

    query_date_obj = None
    if filter_date:
        query_date_obj = _parse_date_string(filter_date, now.year)
        if query_date_obj:
            simple_filters["date"] = filter_date
            simple_filters["day"] = query_date_obj.strftime("%A")
        elif filter_date:
             simple_filters["date"] = filter_date

    if filter_day:
        simple_filters["day"] = filter_day
    if filter_location:
        simple_filters["location"] = filter_location

    if len(simple_filters) == 1:
        k, v = list(simple_filters.items())[0]
        chroma_filter[k] = {"$eq": v}
    elif len(simple_filters) > 1:
        chroma_filter["$or"] = [{k: {"$eq": v}} for k, v in simple_filters.items()]

    kwargs = {"k": k_value}
    if chroma_filter:
        kwargs["filter"] = chroma_filter

    docs = retriever.invoke(search_query, **kwargs)
    if not docs:
        return "I couldn't find any upcoming events matching those criteria."

    filtered = []
    seen = set()

    for doc in docs:
        title = doc.metadata.get('title')
        start_str = str(doc.metadata.get('start_date_meta', '')).strip()
        end_str = str(doc.metadata.get('end_date_meta', '')).strip()
        day_val = str(doc.metadata.get('day', '')).strip()
        time_str = str(doc.metadata.get('time', '')).strip()

        key = (title, start_str, end_str, day_val)
        if key in seen:
            continue
        
        doc_start_date = _parse_date_string(start_str, now.year)
        doc_end_date = _parse_date_string(end_str, now.year)

        # --- Filter A: Strict Date Range Match ---
        if query_date_obj:
            is_match = False
            
            # 1. Check strict date range (if available)
            if doc_start_date and doc_end_date:
                if doc_start_date <= query_date_obj <= doc_end_date:
                    is_match = True
            
            # 2. If no date range, check Day of Week (e.g., "Friday")
            elif day_val:
                query_day_short = query_date_obj.strftime("%a").lower() # e.g., 'fri'
                if query_day_short in day_val.lower():
                    is_match = True
            
            if not is_match:
                continue

        # --- Filter B: Past Event Exclusion (Based on End Date) ---
        if doc_end_date and doc_end_date < today:
            continue 

        # --- Filter C: Time Exclusion (For Events Happening Today) ---
        is_happening_today = False
        if doc_start_date and doc_end_date and (doc_start_date <= today <= doc_end_date):
            is_happening_today = True
        elif not doc_start_date and day_val:
            today_day_name = today.strftime("%A") 
            if day_val.lower() == today_day_name.lower():
                is_happening_today = True
        
        if is_happening_today:
            event_time = parse_time_for_sort(time_str)
            if event_time < now_time:
                continue 

        seen.add(key)
        filtered.append(doc)

    if not filtered:
        return "I couldn't find any upcoming or ongoing events matching those criteria."

    for doc in filtered:
        raw = (doc.metadata.get('category') or "").lower()
        doc.metadata["_sort_time"] = parse_time_for_sort(doc.metadata.get("time", ""))

        if "date" in raw:
            doc.metadata["category"] = "Date-specific Events"
        elif "week" in raw:
            doc.metadata["category"] = "Weekly Events"
        elif "daily" in raw or "appoint" in raw or "everyday" in raw:
            doc.metadata["category"] = "Daily Events"
        else:
            if is_date_specific(doc.metadata.get('date', ''), doc.metadata.get('day', '')):
                doc.metadata["category"] = "Date-specific Events"
            elif doc.metadata.get('day'):
                doc.metadata["category"] = "Weekly Events"
            else:
                doc.metadata["category"] = "Daily Events"

    categories = ["Date-specific Events", "Weekly Events", "Daily Events"]
    buckets = {c: [] for c in categories}

    for d in filtered:
        c = d.metadata.get("category")
        if c in buckets:
            buckets[c].append(d)

    for c in buckets:
        buckets[c].sort(key=lambda d: d.metadata["_sort_time"])

    EVENT_DATA_STORE.clear()
    out = []
    idx = 0

    broad = (specificity.lower() == "broad")

    for c in ["Date-specific Events", "Weekly Events"]:
        if buckets[c]:
            out.append(f"\n **{c}**")
            for d in buckets[c]:
                idx += 1
                EVENT_DATA_STORE[idx] = d
                out.append(format_summary_numbered(idx, d.metadata))

    if not broad and buckets["Daily Events"]:
        out.append("\n## 🌞 Daily Events")
        for d in buckets["Daily Events"]:
            idx += 1
            EVENT_DATA_STORE[idx] = d
            out.append(format_summary_numbered(idx, d.metadata))

    if broad:
        out.append(
            "\nThere are Daily Events also happening every day.\n"
            "👉 <a href='#SHOWDAILY::YES'>Yes</a> "
            "👉 <a href='#SHOWDAILY::NO'>No</a>"
        )

    return "\n".join(out)

def get_event_details_core(identifier: str) -> str:
    global EVENT_DATA_STORE
    
    if identifier is None:
        return "No event identifier provided."

    ident = str(identifier).strip()

    m = re.match(r'details\((\d+)\)', ident)
    if m:
        num = int(m.group(1))
    elif ident.isdigit():
        num = int(ident)
    else:
        return "I could not parse that identifier. Use `3` or `details(3)`."

    doc = EVENT_DATA_STORE.get(num)
    if not doc:
        return f"I could not find an event numbered {num}."

    return f"{num}. {format_event_card(doc.metadata, doc.page_content)}"

@function_tool
def get_event_details(identifier: str) -> str:
    """Returns the full details for a specific event."""
    return get_event_details_core(identifier)

INSTRUCTIONS = f"""
You are an AI assistant that answers user questions about Auroville events.

Your job:
- Handle general queries such as:
    "events tomorrow",
    "yoga events",
    "workshops this weekend",
    "events on 24 November",
    "sound healing",
    "children events",
    "what is happening today",
    "things to do in Auroville", etc.

- For general search queries:
      1) First call `vectordb_query_selector_agent` to refine the search query.
      2) Then call `search_auroville_events` using the refined query.
      3) You may correct small grammar issues *outside* the numbered blocks or remove duplicate text for better understanding.
      4) When tool output lists categories, keep them exactly:
         - Date-specific Events
         - Weekly Events
         - Daily Events

- You MUST NOT handle:
      • details(NUM)
      • NUM  
      • show daily events  
  because they are handled directly by the application code.  
  Simply treat them as normal text when streamed to you — but you will never be asked to respond to them because the app intercepts them before you are invoked.

Rules while formatting responses:
1. **Do not change numbering** of event summaries.
2. **Do not rewrite or remove summary lines generated by the tools.**
3. **Do not modify or break clickable HTML links** (e.g., View Details, Yes/No).  
   These links must remain untouched.
4. ** You MUST remove Duplicate events.

When a tool is called:
- You must simply return the tool output to the user without reformatting.

Do not hallucinate missing event information.  
If metadata is missing, omit that field.

"""

tools = [
    vectordb_query_selector_agent.as_tool("vectordb_query_selector_agent", "Refines query."),
    search_auroville_events,
    get_daily_events,
    get_event_details
]

auroville_agent = Agent(
    name="Auroville Events Assistant",
    instructions=INSTRUCTIONS,
    model=gemini_model,
    tools=tools
    )
