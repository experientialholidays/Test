import os
import logging
import urllib.parse
from datetime import datetime, time
from typing import Optional, Dict, Any, List
import re

from agents import Agent, function_tool, OpenAIChatCompletionsModel
from vector_db import VectorDBManager
from vectordb_query_selector_agent import vectordb_query_selector_agent
from openai import AsyncOpenAI
from langchain_core.documents import Document

# ... [Keep your Date/Time Helper Functions: is_date_specific, parse_time_for_sort] ...
# (Paste your existing helper functions here or keep them if you are editing the file)

def is_date_specific(date_str, day_str):
    return bool(date_str and str(date_str).strip().lower() not in ('', 'n/a', 'upcoming', 'none'))

def parse_time_for_sort(raw: str) -> time:
    if not raw: return time(23, 59, 59)
    s = str(raw).replace("—", "-").replace("–", "-").strip().upper()
    if re.search(r'\bANYTIME\b|\bOPEN\b|\bALL DAY\b', s): return time(23, 59, 59)
    token_pattern = re.compile(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?')
    tokens = list(token_pattern.finditer(s))
    if not tokens:
        trailing = re.search(r'(\d{1,2})(?::(\d{2}))?[-\s]*\d{1,2}(?::\d{2})?\s*(AM|PM)', s)
        if trailing:
            h = int(trailing.group(1))
            m = int(trailing.group(2)) if trailing.group(2) else 0
            mer = trailing.group(3)
            if mer == "PM" and h != 12: h += 12
            if mer == "AM" and h == 12: h = 0
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
    if mer == "PM" and hour != 12: hour += 12
    if mer == "AM" and hour == 12: hour = 0
    return time(hour, minute)

# -------------------------------------------------------------------------
# Setup & Global Cache
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVENT_DATA_STORE: Dict[int, Document] = {}

VECTOR_DB_NAME = "vector_db"
DB_FOLDER = "input"
MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv('GOOGLE_API_KEY')

db_manager = VectorDBManager(folder=DB_FOLDER, db_name=VECTOR_DB_NAME)
retriever = None

def initialize_retriever(vectorstore):
    global retriever
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
    else:
        logger.error("Retriever init failed: Vectorstore is None.")

vectorstore = db_manager.create_or_load_db()
initialize_retriever(vectorstore)

gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model=MODEL, openai_client=gemini_client)

# ... [Keep your Formatting Helpers: format_event_card, format_summary_numbered] ...
# (Paste your existing formatting functions here)

def format_event_card(doc_metadata: Dict, doc_content: str) -> str:
    title = doc_metadata.get('title', 'Event').strip()
    date_str = doc_metadata.get('date', '').strip()
    day_str = doc_metadata.get('day', '').strip()
    time_str = doc_metadata.get('time', '').strip()
    location = doc_metadata.get('location', '').strip()
    contribution = doc_metadata.get('contribution', '').strip()
    contact_info = doc_metadata.get('contact', '').strip()
    description = doc_content.strip()
    poster_url = doc_metadata.get('poster_url')
    phone_number = doc_metadata.get('phone', '').strip()
    category = doc_metadata.get('category', '').strip()
    out = []
    out.append(f"**Event Name:** {title}")
    if category: out.append(f"**Category:** {category}")
    when = []
    if day_str: when.append(day_str)
    if date_str: when.append(date_str)
    if time_str: when.append(f"@ {time_str}")
    if when: out.append(f"**When:** {' '.join(when)}")
    if location: out.append(f"**Where:** {location}")
    if contribution: out.append(f"**Contribution:** {contribution}")
    clean_phone = ''.join(filter(str.isdigit, phone_number))
    wa_link = ""
    if clean_phone:
        msg = f"Hi, I came across your event '{title}' scheduled on {date_str}. Info?"
        wa = urllib.parse.quote(msg)
        wa_link = f"\n[**Click to Chat on WhatsApp**](https://wa.me/{clean_phone}?text={wa})"
    if contact_info or clean_phone: out.append(f"**Contact:** {contact_info}{wa_link}")
    out.append("\n**Description:**")
    out.append(description)
    if poster_url: out.append(f"\n\n![Event Poster]({poster_url})")
    return "\n".join(out)

def format_summary_numbered(index: int, meta: Dict) -> str:
    title = meta.get('title', '').strip()
    day = meta.get('day', '').strip()
    time_str = meta.get('time', '').strip()
    loc = meta.get('location', '').strip()
    contrib = meta.get('contribution', '').strip()
    phone = meta.get('phone', '').strip()
    parts = []
    if day: parts.append(day)
    if time_str: parts.append(time_str)
    if loc: parts.append(f"@{loc}")
    if contrib: parts.append(f"| Contrib: {contrib}")
    if phone: parts.append(f"| Ph:{''.join(filter(str.isdigit, phone))}")
    line = " ".join(parts)
    return (
        f"{index}. **{title}** — {line}\n"
        f"   👉 <a href='#DETAILS::{index}'>View details **({index})**</a>"
    )

# -------------------------------------------------------------------------
# TOOLS - REFACTORED (CORE FUNCTIONS + WRAPPERS)
# -------------------------------------------------------------------------

# 1. DAILY EVENTS CORE (Callable by app.py)
def get_daily_events_core(start_number: int) -> str:
    global EVENT_DATA_STORE, vectorstore
    try:
        raw = vectorstore.get(where={"category": "Daily Events"}, include=["documents", "metadatas"])
    except Exception as e:
        return f"Error fetching daily events: {e}"

    if not raw or not raw.get("ids"):
        return "No Daily Events found."

    docs = [Document(page_content=text, metadata=meta) for text, meta in zip(raw["documents"], raw["metadatas"])]
    for d in docs: d.metadata["_sort_time"] = parse_time_for_sort(d.metadata.get("time", ""))
    docs.sort(key=lambda d: d.metadata["_sort_time"])
    out = ["\n## 🌞 Daily Events"]
    idx = int(start_number)
    for d in docs:
        idx += 1
        EVENT_DATA_STORE[idx] = d
        out.append(format_summary_numbered(idx, d.metadata))
    return "\n".join(out)

# 1. DAILY EVENTS TOOL (Callable by Agent)
@function_tool
def get_daily_events(start_number: int) -> str:
    """Returns ALL Daily Events using metadata filtering only."""
    return get_daily_events_core(start_number)


# 2. SEARCH TOOL (Unchanged, already a tool)
@function_tool
def search_auroville_events(search_query: str, specificity: str, filter_day: Optional[str] = None, filter_date: Optional[str] = None, filter_location: Optional[str] = None) -> str:
    # ... (Paste the full body of your original search_auroville_events here) ...
    # (I am omitting the body for brevity, but you must keep the original logic intact)
    global retriever, EVENT_DATA_STORE
    if retriever is None: return "The event database is still initializing."
    # ... [Keep original search logic] ...
    return "I couldn't find any events." # (Placeholder - keep your real code)


# 3. EVENT DETAILS CORE (Callable by app.py)
def get_event_details_core(identifier: str) -> str:
    global EVENT_DATA_STORE
    if identifier is None: return "No event identifier provided."
    ident = str(identifier).strip()
    m = re.match(r'details\((\d+)\)', ident)
    if m: num = int(m.group(1))
    elif ident.isdigit(): num = int(ident)
    else: return "I could not parse that identifier. Use `3` or `details(3)`."
    doc = EVENT_DATA_STORE.get(num)
    if not doc: return f"I could not find an event numbered {num}."
    return f"{num}. {format_event_card(doc.metadata, doc.page_content)}"

# 3. EVENT DETAILS TOOL (Callable by Agent)
@function_tool
def get_event_details(identifier: str) -> str:
    return get_event_details_core(identifier)


# -------------------------------------------------------------------------
# Agent Setup
# -------------------------------------------------------------------------
INSTRUCTIONS = f"""
You are an AI assistant for Auroville events.
(Keep your original instructions here)
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
