# auroville_agent.py
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

# -------------------------------------------------------------------------
# Date/Time Helper Functions
# -------------------------------------------------------------------------

def is_date_specific(date_str, day_str):
    """Classifies an event as date-specific."""
    return bool(date_str and str(date_str).strip().lower() not in ('', 'n/a', 'upcoming', 'none'))

# -------------------------------------------------------------------------
# Robust Time Parser for Sorting
# -------------------------------------------------------------------------

def parse_time_for_sort(raw: str) -> time:
    if not raw:
        return time(23, 59, 59)

    s = str(raw).replace("—", "-").replace("–", "-").strip().upper()

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

# Initialize DB & retriever eagerly (same as original file)
vectorstore = db_manager.create_or_load_db()
initialize_retriever(vectorstore)

gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model=MODEL, openai_client=gemini_client)

# -------------------------------------------------------------------------
# 2. Formatting Helpers
# -------------------------------------------------------------------------

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

    clean_phone = ''.join(filter(str.isdigit, phone_number))
    wa_link = ""
    if clean_phone:
        msg = f"Hi, I came across your event '{title}' scheduled on {date_str}. Info?"
        wa = urllib.parse.quote(msg)
        wa_link = f"\n[**Click to Chat on WhatsApp**](https://wa.me/{clean_phone}?text={wa})"

    if contact_info or clean_phone:
        out.append(f"**Contact:** {contact_info}{wa_link}")

    out.append("\n**Description:**")
    out.append(description)

    if poster_url:
        out.append(f"\n\n![Event Poster]({poster_url})")

    return "\n".join(out)

# -------------------------------------------------------------------------
# Clickable, numbered summary formatting (no FETCH links)
# -------------------------------------------------------------------------

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
    if phone:
        parts.append(f"| Ph:{''.join(filter(str.isdigit, phone))}")

    line = " ".join(parts)
    # numbered line plus a plain anchor that the frontend converts into a code or click
    return (
        f"{index}. **{title}** — {line}\n"
        f"   👉 ` {index} `  — click or type the number to fetch details"
    )

# -------------------------------------------------------------------------
# 3. Tools
# -------------------------------------------------------------------------

@function_tool
def get_daily_events(start_number: int) -> str:
    """
    Returns ALL Daily Events using metadata filtering only.
    Category = 'Daily Events'
    Numbering continues from start_number (so caller can append)
    """
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

# -------------------------------------------------------------------------
# UPDATED search_auroville_events()
# (keeps previous behavior; removed any mention of FETCH links)
# -------------------------------------------------------------------------

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

    # date/day normalization
    if filter_date:
        simple_filters["date"] = filter_date
        try:
            for fmt in ["%B %d, %Y", "%B %d", "%d %B", "%d %b", "%d.%m.%y", "%d.%m.%Y"]:
                try:
                    year = datetime.now().year
                    parse_str = filter_date
                    if "%Y" not in fmt and "%y" not in fmt:
                        parse_str = f"{filter_date}, {year}"
                    dt = datetime.strptime(parse_str.strip(), fmt.strip())
                    simple_filters["day"] = dt.strftime("%A")
                    break
                except:
                    continue
        except:
            pass

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

    now = datetime.now()
    today = now.date()
    now_time = now.time()

    filtered = []
    seen = set()

    for doc in docs:
        title = doc.metadata.get('title')
        date_str = str(doc.metadata.get('date', '')).strip()
        day_val = str(doc.metadata.get('day', '')).strip()
        time_str = str(doc.metadata.get('time', '')).strip()

        key = (title, date_str, day_val)
        if key in seen:
            continue

        if is_date_specific(date_str, day_val):
            try:
                event_dt = None
                for fmt in ["%B %d, %Y", "%B %d", "%d %B", "%d %b", "%d.%m.%y", "%d.%m.%Y"]:
                    try:
                        p = date_str
                        if "%Y" not in fmt and "%y" not in fmt:
                            p = f"{date_str}, {today.year}"
                        d = datetime.strptime(p.strip(), fmt.strip()).date()
                        t = parse_time_for_sort(time_str)
                        event_dt = datetime.combine(d, t)
                        break
                    except:
                        continue
                if event_dt:
                    if event_dt.date() < today:
                        continue
                    if event_dt.date() == today and event_dt.time() < now_time:
                        continue
            except:
                pass

        seen.add(key)
        filtered.append(doc)

    if not filtered:
        return "I couldn't find any upcoming or ongoing events matching those criteria."

    # Normalize categories
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

    # Show normal categories first
    for c in ["Date-specific Events", "Weekly Events"]:
        if buckets[c]:
            out.append(f"\n## 📅 **{c}**")
            for d in buckets[c]:
                idx += 1
                EVENT_DATA_STORE[idx] = d
                out.append(format_summary_numbered(idx, d.metadata))

    # Specific search → show daily events normally
    if not broad and buckets["Daily Events"]:
        out.append("\n## 🌞 Daily Events")
        for d in buckets["Daily Events"]:
            idx += 1
            EVENT_DATA_STORE[idx] = d
            out.append(format_summary_numbered(idx, d.metadata))

    # Broad search → hide + ask Yes/No
    if broad:
        out.append(
            "\nThere are Daily Events also happening every day.\n"
            "👉 <a href='#SHOWDAILY::YES'>Yes</a> "
            "👉 <a href='#SHOWDAILY::NO'>No</a>"
        )

    return "\n".join(out)

# -------------------------------------------------------------------------
# get_event_details (unchanged behavior — returns cached card)
# -------------------------------------------------------------------------

@function_tool
def get_event_details(identifier: str) -> str:
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

# -------------------------------------------------------------------------
# 4. Agent Instructions — Revised and minimal (no FETCH mention)
# -------------------------------------------------------------------------

INSTRUCTIONS = f"""
You are an AI Event Information Extractor for Auroville.

Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.
Set temperature to 0.1.

Primary responsibilities:
- Answer general user queries about events (e.g., "events tomorrow", "what's on 24 November").
  For these queries, call the selector and the search tool workflows:
    1) Use `vectordb_query_selector_agent` to refine the user's query when appropriate.
    2) Call `search_auroville_events` with the refined query and the appropriate specificity.
    3) Present the exact returned tool output to the user after fixing minor grammar or removing accidental duplicate lines. 
       **Do not modify the numbered summary tokens** generated by the tool (they are used by the frontend/caller to fetch details).

Special rules (enforced by the application code or frontend):
- If the user's input is a plain integer (e.g., `4`) or matches `details(NUM)` (e.g., `details(4)`), do NOT perform a new vector DB search. 
  Instead, the application code (frontend/server) will call the tool `get_event_details` directly and return its output; the LLM should not attempt to re-route or re-run searches for those requests.
- If the user requests "show daily events" (or clicks the daily events confirmation), the application code will call `get_daily_events(start_number)` directly and return its output; the LLM should not invoke the selector or search for this step.
- In other words: clicking/view-details or show-daily is handled by the application logic; the assistant need not decide routing for these events.

Display rules:
- Numbered entries in results must remain exactly as produced by the search tool; the assistant may only correct grammar outside those numbered entries.
- When returning details from `get_event_details`, do not perform any additional search or re-ranking — return the cached event card as-is.
- Do not hallucinate missing metadata — omit missing fields.
- Keep event categories: Date-specific Events, Weekly Events, Daily Events.

Workflow exception:
- If the user's question is exactly "daily and appointment-based events", the application will call:
    search_auroville_events(search_query="daily and appointment-based events", specificity="Broad")
  (this is handled by the application; the assistant may supply the search term but should not override the app's routing.)

Always follow the rules above.
"""

# -------------------------------------------------------------------------
# Tools & Agent setup (no FETCH tool included)
# -------------------------------------------------------------------------

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
