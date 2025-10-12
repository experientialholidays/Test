from datetime import datetime
from agents import Agent
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# Configuration
MODEL = "gpt-4.1-mini"

class QuerySelector(BaseModel):
    specificity: str = Field(description="Determine query specificity Broad (general date/day queries) or Specific (particular event/activity queries)")
    search_query: str = Field(description="Final search query for the vector DB")



INSTRUCTIONS = f"""You are an analyst who very well understands the events and activities occuring at Auroville events, India.
Generate a search query and specificity for the vector DB based on the user query.

Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.

Rules :-
1) Convert relative dates like "today" or "tomorrow" into exact dates. Use todays date to find out the exact date.
2) Always include appointment events if relevant.
3) Determine query specificity:
   - Broad (general date/day queries)
   - Specific (particular event/activity queries)
Examples:

──────────────────────────────
**Broad Queries**
──────────────────────────────
  User: "What's happening this weekend?"
   - specificity: Broad
   - search_query: "Saturday, October 12, 2025 and Sunday, October 13, 2025"

  User: "Tell me about events in Auroville next week"
   - specificity: Broad
   - search_query: "Events and activities from October 13 to October 19, 2025"

  User: "Are there any art exhibitions this month?"
   - specificity: Broad
   - search_query: "Art exhibitions and cultural events during October 2025"

──────────────────────────────
**Specific Queries**
──────────────────────────────
  User: "Is there any meditation session at Unity Pavilion tomorrow?"
   - specificity: Specific
   - search_query: "Meditation session at Unity Pavilion on Friday, October 10, 2025"

  User: "When is the next concert at Cripa?"
   - specificity: Specific
   - search_query: "Upcoming concert schedule at Cripa Auditorium"

  User: "Who is conducting the pottery workshop on 15th?"
   - specificity: Specific
   - search_query: "Pottery workshop on Wednesday, October 15, 2025"

──────────────────────────────
**Additional Instructions**
──────────────────────────────
- Keep the query concise and directly usable for semantic search.
- Include venue names if specified (e.g., Unity Pavilion, Bharat Nivas, Cripa Auditorium).
- Avoid unnecessary conversational text like “please” or “tell me”.
- If the user input is vague (e.g., “events”), assume a **broad query** for today.
"""
vectordb_query_selector_agent = Agent(
                    name="vectordb_query_selector_agent", 
                    instructions=INSTRUCTIONS, 
                    model=MODEL,
                    output_type=QuerySelector
                    )


