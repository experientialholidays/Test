import os
import logging
import urllib.parse
import ast
from datetime import datetime, time
from typing import Optional, Dict, Any, List
import re  # added for better time parsing

from agents import Agent, function_tool, OpenAIChatCompletionsModel
from vector_db import VectorDBManager
from vectordb_query_selector_agent import vectordb_query_selector_agent
from openai import AsyncOpenAI
from langchain_core.documents import Document

# -------------------------------------------------------------------------
# Date/Time Helper Functions
# -------------------------------------------------------------------------

def is_date_specific(date_str, day_str):
    return bool(date_str and date_str.lower() not in ('', 'n/a', 'upcoming', 'none'))

# -------------------------------------------------------------------------
# FIX #1 — Improved Time Parser (sorting only)
# -------------------------------------------------------------------------
def parse_time_for_sort(raw: str) -> time:
    """Extracts the first valid time in a messy time string for sorting."""
    if not raw:
        return time(23, 59, 59)

    s = raw.replace("—", "-").replace("–", "-").strip().upper()

    # Look for first time-like item
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?", s)
    if not match:
        return time(23, 59, 59)

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    mer = match.group(3)

    if mer == "PM" and hour != 12:
        hour += 12
    if mer == "AM" and hour == 12:
        hour = 0

    return time(hour, minute)

# -------------------------------------------------------------------------
# 1. Setup & Global Cache
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

# lazy init
retriever = None

def initialize_retriever(vectorstore):
    global retriever
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
        logger.info("Auroville agent retriever successfully initialized.")
    else:
        logger.error("Failed to initialize retriever: Vectorstore is None.")

# ---------------- REQUIRED ORDER FIX ----------------
vectorstore = db_manager.create_or_load_db()
initialize_retriever(vectorstore)

# Model
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
    poster_url = doc_metadata.get('poster_url', None)
    phone_number = doc_metadata.get('phone', '').strip()
    category = doc_metadata.get('category', '').strip()

    output = []
    output.append(f"**Event Name:** {title}")

    if category:
        output.append(f"**Category:** {category}")

    when = []
    if day_str: when.append(day_str)
    if date_str: when.append(date_str)
    if time_str: when.append(f"@ {time_str}")

    if when:
        output.append(f"**When:** {' '.join(when)}")

    if location:
        output.append(f"**Where:** {location}")

    if contribution:
        output.append(f"**Contribution:** {contribution}")

    clean_phone = ''.join(filter(str.isdigit, phone_number))
    wa_link = ""
    if clean_phone:
        msg = f"Hi, I came across your event '{title}' scheduled on {date_str}. Info?"
        encoded_msg = urllib.parse.quote(msg)
        wa_link = f"\n[**Click to Chat on WhatsApp**](https://wa.me/{clean_phone}?text={encoded_msg})"

    if contact_info or clean_phone:
        output.append(f"**Contact:** {contact_info}{wa_link}")

    output.append("\n**Description:**")
    output.append(description)

    if poster_url:
        output.append(f"\n\n![Event Poster]({poster_url})")

    return "\n".join(filter(None, output))


def format_summary_line(doc_metadata: Dict) -> str:
    title = doc_metadata.get('title', 'Event').strip()
    day = doc_metadata.get('day', '').strip()
    time_str = doc_metadata.get('time', '').strip()
    loc = doc_metadata.get('location', '').strip()
    contribution = doc_metadata.get('contribution', '').strip()
    phone_number = doc_metadata.get('phone', '').strip()
    category = doc_metadata.get('category', '').strip()

    safe_key = urllib.parse.quote(title)
    summary = [f"- [**{title}**](#FETCH::{safe_key})"]

    details = []
    if category: details.append(f"({category})")
    if day: details.append(day)
    if time_str: details.append(time_str)
    if loc: details.append(f"@{loc}")
    if contribution: details.append(f"| Contribution: {contribution}")

    clean_phone = ''.join(filter(str.isdigit, phone_number))
    if clean_phone:
        details.append(f"| Ph: {clean_phone}")

    if details:
        summary.append("|")
        summary.append(" ".join(details))

    return " ".join(summary)

# -------------------------------------------------------------------------
# 3. search_auroville_events tool
# -------------------------------------------------------------------------

@function_tool
def search_auroville_events(
    search_query: str,
    specificity: str,
    filter_day: Optional[str] = None,
    filter_date: Optional[str] = None,
    filter_location: Optional[str] = None
) -> str:

    global retriever
    if retriever is None:
        return "The event database is still initializing. Please wait a moment."

    k_value = 100 if specificity.lower() == "broad" else 12
    chroma_filter = {}
    simple_filters = {}

    if filter_date:
        simple_filters["date"] = filter_date
        try:
            for fmt in ["%B %d, %Y", "%B %d", "%d %B", "%d %b", "%d.%m.%y", "%d.%m.%Y"]:
                try:
                    current_year = datetime.now().year
                    parse_str = filter_date
                    if "%Y" not in fmt and "%y" not in fmt:
                        parse_str = f"{filter_date}, {current_year}"
                    dt = datetime.strptime(parse_str.strip(), fmt.strip())
                    simple_filters["day"] = dt.strftime("%A")
                    break
                except: continue
        except: pass

    if filter_day: simple_filters["day"] = filter_day
    if filter_location: simple_filters["location"] = filter_location

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

    # Remove past events
    now = datetime.now()
    today = now.date()
    now_t = now.time()

    filtered = []
    for doc in docs:
        date_str = doc.metadata.get('date', '').strip()
        time_str = doc.metadata.get('time', '').strip()
        if is_date_specific(date_str, doc.metadata.get('day', '')):
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
                    except: continue

                if event_dt:
                    if event_dt.date() < today: continue
                    if event_dt.date() == today and event_dt.time() < now_t: continue
            except:
                pass
        filtered.append(doc)

    if not filtered:
        return "I couldn't find any upcoming or ongoing events matching those criteria."

    # Sort
    for doc in filtered:
        doc.metadata["_sort_time"] = parse_time_for_sort(doc.metadata.get("time", ""))

    category_order = {
        "Date-specific": 1,
        "Weekday-based": 2,
        "Appointment/Daily-based": 3
    }

    def sort_key(doc):
        return (
            category_order.get(doc.metadata.get("category", ""), 4),
            doc.metadata["_sort_time"]
        )

    filtered.sort(key=sort_key)

    EVENT_DATA_STORE.clear()
    final = []
    seen = set()

    for doc in filtered:
        key = (doc.metadata.get("title"), doc.metadata.get("date"), doc.metadata.get("day"))
        safe = urllib.parse.quote(doc.metadata.get("title", ""))
        EVENT_DATA_STORE[safe] = doc
        if key not in seen:
            seen.add(key)
            final.append(doc)

    out = []
    count = len(final)
    is_exact = count > 0 and search_query.lower() in final[0].metadata.get("title", "").lower()

    if count < 5 or specificity.lower() == "specific" or is_exact:
        out.append(f"Found {count} event(s):\n")
        for d in final:
            out.append(format_event_card(d.metadata, d.page_content))
            out.append("\n" + "=" * 30 + "\n")
    else:
        out.append(f"Found {count} events. Click a name to see details:\n")
        current_cat = None
        for d in final:
            category = d.metadata.get("category", "Other Events") or "Other Events"
            if category != current_cat:
                out.append(f"\n## 📅 **{category}**")
                current_cat = category
            out.append(format_summary_line(d.metadata))

        if specificity.lower() == "broad":
            out.append("\n\n[**See daily & appointment-based events**](#TRIGGER_SEARCH::daily and appointment-based events::Broad)")

    return "\n".join(out)

# -------------------------------------------------------------------------
# 4. Agent Instructions
# -------------------------------------------------------------------------

INSTRUCTIONS = f"""
You are an **AI Event Information Extractor** dedicated to providing structured and accurate event details from the Auroville community.

Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.

Set your temperature **0.1**.

Your role is to help users find information about events, activities, workshops, and schedules.

You have access to two tools:
1) **`vectordb_query_selector_agent`**: Generates the best possible refined search query and specificity.
2) **`search_auroville_events`**: Searches the vector database and filters results.

### **Event Rules Applied by Search Tool**
* Shows only **upcoming or ongoing** events.
* Groups events by **Category**.
* Sorts events **chronologically by time** inside categories.
* Never hallucinate event details.

### **Workflow**
1. If user asks *exactly* “daily and appointment-based events”, skip the selector tool.
2. Otherwise call `vectordb_query_selector_agent`.
3. Then call `search_auroville_events`.
4. After receiving the tool output, clean up grammar and formatting BUT:

### ❗ FIX #2 — CRITICAL RULE:
**IMPORTANT: Do NOT modify or remove any Markdown links (such as [**Event Name**](#FETCH::key)).  
Leave all links exactly as generated. Never rewrite them.**

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
