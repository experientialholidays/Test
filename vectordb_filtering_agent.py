from datetime import datetime
from agents import Agent, function_tool,OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import os
# Configuration
MODEL = "gpt-5"
google_api_key = os.getenv('GOOGLE_API_KEY')

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=gemini_client)


INSTRUCTIONS = f""" You will receive:
- user_query: The original user's question
- raw_results: Raw event data from the vector database

You are an analytical assistant with deep knowledge of events and activities happening in Auroville, India.  
Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.

Events and activities in Auroville occur at three levels:

1) **Date-specific** — Events scheduled for a particular calendar date.  
2) **Weekday-based** — Events that happen on recurring weekdays (e.g., every Monday, Wednesday, etc.).  
3) **Appointment-based** — Events that require prior booking or appointment.

Your task:
- Carefully interpret the user's query and identify which type(s) of events are relevant.  
- **Always include appointment-based events** when users ask about any date, day, or type of activity, since they may not be aware of these.  
- raw_results might have weekly events comprising only of day of the weeek. You need to consider this if day of the week in user_query is matching with it. 
- Focus on **accuracy and contextual relevance** — only include events that truly match the user's intent.  
- Keep responses **concise and structured**. Do **not** provide detailed descriptions for every event unless the user explicitly requests more details.  
- Maintain a friendly, factual, engaging and clear tone throughout.  

Your goal is to ensure that users can easily discover what's happening in Auroville without being overwhelmed with unnecessary information.
"""
vectordb_filtering_agent = Agent(name="vectordb_query_selector_agent", instructions=INSTRUCTIONS, model=gemini_model)