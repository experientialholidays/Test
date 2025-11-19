import os
import logging
import urllib.parse
import ast
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
# Robust Time Parser for Sorting (improved)
# -------------------------------------------------------------------------
def parse_time_for_sort(raw: str) -> time:
    """
    Extracts the first valid time in a messy time string for sorting.
    Handles ranges with dashes, multiple slots, AM/PM at end of range,
    missing minutes, unicode dashes, and simple textual labels.
    Returns a time object (defaults to 23:59:59 if unparseable).
    """
    if not raw:
        return time(23, 59, 59)

    s = str(raw).replace("—", "-").replace("–", "-").strip().upper()

    # If clearly non-specific
    if re.search(r'\bANYTIME\b|\bOPEN\b|\bALL DAY\b', s):
        return time(23, 59, 59)

    # Find tokens like "8:30 AM", "8 AM", "8", etc.
    token_pattern = re.compile(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?')
    tokens = list(token_pattern.finditer(s))

    if not tokens:
        # Try pattern "7-9 AM" (AM/PM trailing)
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

    # Prefer first token that has AM/PM. Otherwise take first token and try to infer meridian.
    chosen = None
    mer = None
    for t in tokens:
        if t.group(3):
            chosen = t
            mer = t.group(3)
            break

    if not chosen:
        chosen = tokens[0]
        # Look ahead for AM/PM in a short window after token (e.g., "7-9 AM")
        after = s[chosen.end(): chosen.end() + 10]
        m_after = re.search(r'\b(AM|PM)\b', after)
        mer = m_after.group(1) if m_after else None

    hour = int(chosen.group(1))
    minute = int(chosen.group(2)) if chosen.group(2) else 0

    if mer == "PM" and hour != 12:
        hour += 12
    if mer == "AM" and hour == 12:
        hour = 0

    # Clamp sanity
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
    """
    Formats the summary line with an INTERNAL fetch link.
    IMPORTANT: link must be a fragment #FETCH::key so frontend can intercept it.
    """
    title = doc_metadata.get('title', 'Event').strip()
    day = doc_metadata.get('day', '').strip()
    time_str = doc_metadata.get('time', '').strip()
    loc = doc_metadata.get('location', '').strip()
    contribution = doc_metadata.get('contribution', '').strip()
    phone_number = doc_metadata.get('phone', '').strip()
    category = doc_metadata.get('category', '').strip()

    safe_key = urllib.parse.quote(title, safe='')
    link_target = f"#FETCH::{safe_key}"

    summary = [f"- [**{title}**]({link_target})"]

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
        logger.error("Retriever is not initialized. Cannot perform search.")
        return "The event database is still initializing. Please try again."

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

    # Execute search (retriever may be LangChain/Chroma retriever)
    docs = retriever.invoke(search_query, **kwargs)

    if not docs:
        return "I couldn't find any upcoming events matching those criteria."

    # --- FILTER OUT PAST EVENTS & DEDUPE ---
    now = datetime.now()
    today = now.date()
    now_t = now.time()

    filtered = []
    seen = set()  # to dedupe exact duplicates by (title,date,day)
    for doc in docs:
        title = doc.metadata.get('title')
        date_str = (doc.metadata.get('date') or "").strip()
        day_val = (doc.metadata.get('day') or "").strip()
        time_str = (doc.metadata.get('time') or "").strip()

        dedup_key = (title, date_str, day_val)
        if dedup_key in seen:
            continue

        # Exclude past date-specific events
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
                    if event_dt.date() < today:
                        continue
                    if event_dt.date() == today and event_dt.time() < now_t:
                        continue
            except:
                # if parsing fails, keep the event for safety
                pass

        seen.add(dedup_key)
        filtered.append(doc)

    if not filtered:
        return "I couldn't find any upcoming or ongoing events matching those criteria."

    # --- NORMALIZE CATEGORY and assign sort-time ---
    for doc in filtered:
        doc.metadata["_sort_time"] = parse_time_for_sort(doc.metadata.get("time", ""))

        # Normalize category names for consistent grouping/sorting
        raw_cat = (doc.metadata.get("category") or "").strip().lower()
        if "date" in raw_cat:
            doc.metadata["category"] = "Date-specific Events"
        elif "week" in raw_cat or "weekday" in raw_cat:
            doc.metadata["category"] = "Weekly Events"
        elif "appoint" in raw_cat or "daily" in raw_cat or "everyday" in raw_cat:
            doc.metadata["category"] = "Daily / Appointment-based Events"
        else:
            # infer if blank
            if is_date_specific(doc.metadata.get('date', ''), doc.metadata.get('day', '')):
                doc.metadata["category"] = "Date-specific Events"
            elif doc.metadata.get('day'):
                doc.metadata["category"] = "Weekly Events"
            else:
                doc.metadata["category"] = "Daily / Appointment-based Events"

    # --- BUCKETS: Category -> list of docs ---
    categories = ["Date-specific Events", "Weekly Events", "Daily / Appointment-based Events"]
    category_buckets: Dict[str, List[Document]] = {cat: [] for cat in categories}

    for doc in filtered:
        cat = doc.metadata.get("category", "Daily / Appointment-based Events")
        if cat not in category_buckets:
            category_buckets.setdefault(cat, [])
        category_buckets[cat].append(doc)

    # Sort docs within each category by _sort_time (earliest first)
    for cat in category_buckets:
        category_buckets[cat].sort(key=lambda d: d.metadata.get("_sort_time", time(23, 59, 59)))

    # Build EVENT_DATA_STORE (one key per event) and prepare output lines
    EVENT_DATA_STORE.clear()
    out_lines = []
    total_events = 0

    for cat in categories:
        bucket = category_buckets.get(cat, [])
        if not bucket:
            continue

        out_lines.append(f"\n## 📅 **{cat}**")
        for d in bucket:
            # Each event remains separate; create a safe key per event title
            title = d.metadata.get('title', 'Event')
            safe_key = urllib.parse.quote(title, safe='')

            # Ensure key uniqueness: if collision (same title), append an index
            base_key = safe_key
            idx = 1
            while safe_key in EVENT_DATA_STORE:
                safe_key = urllib.parse.quote(f"{title}::{idx}", safe='')
                idx += 1

            EVENT_DATA_STORE[safe_key] = d

            # Use the summary formatter (preserves internal fragment link)
            out_lines.append(format_summary_line(d.metadata))

            total_events += 1

    header = f"Found {total_events} event(s). Click a name to see details:\n"
    final_output = header + "\n".join(out_lines)

    return final_output

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
* Groups events into these categories: **Date-specific Events**, **Weekly Events**, **Daily / Appointment-based Events**.
* Keeps each event separate (no merging by contact). Each event has its own FETCH link that maps to a stored Document in the backend.
* Sorts events inside each category chronologically by the earliest parsable time in the `time` field.
* Never hallucinate event details.

### **Workflow**
1. If the user's question is exactly "daily and appointment-based events", skip the selector tool and call `search_auroville_events` with search_query="daily and appointment-based events", specificity="Broad".
2. Otherwise call `vectordb_query_selector_agent` to refine the search query and specificity, then call `search_auroville_events`.
3. After receiving the tool output, you MAY fix small grammar issues and remove accidental duplicated lines, but:

### ABSOLUTE RULES (CRITICAL)
* DO NOT modify, reformat, rewrite or convert ANY Markdown fetch link. Links must remain exactly as produced by the tool, in this form: `[**Event Title**](#FETCH::EncodedKey)`.
* DO NOT convert fetch fragments into absolute URLs (do NOT add https://host/...).
* DO NOT change any text inside the square brackets `[...]` or inside the parentheses `( ... )` of fetch links.
* You may only clean the plain text outside of those links (fix grammar, remove accidental repeated phrases, remove duplicate lines), but leave the link tokens untouched.
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
