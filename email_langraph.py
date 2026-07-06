# python -m mlx_lm.server --model jedisct1/gemma-4-E2B-it-txt-mlx-4bit --port 8080
import os
import time
import random
import streamlit as st
from langgraph.graph import START, END, StateGraph
from openai import OpenAI
from typing_extensions import TypedDict
from basic_email import get_emails, get_calendar
from vector_storage import vector_store

# Configure Streamlit layout with two columns: main content and logs
st.set_page_config(layout="wide")
main_content, log_content = st.columns(2)

import re
import datetime

def parse_natural_language_date(query: str):
    """Converts phrases like 'last 2 week summary' into ISO 8601 timestamps."""
    now = datetime.datetime.now(datetime.timezone.utc)
    query_clean = query.lower()

    # Look for "last X week(s)"
    match = re.search(r'last\s+(\d+)\s+week', query_clean)
    if match:
        weeks_count = int(match.group(1))
        time_min = now - datetime.timedelta(weeks=weeks_count)
        return time_min.isoformat(), now.isoformat()

    # Look for a simple "last week"
    if "last week" in query_clean:
        time_min = now - datetime.timedelta(weeks=1)
        return time_min.isoformat(), now.isoformat()

    # Fallback to None if it doesn't match, letting your function use its default
    return None, None

def util_st_log(content):
    log_content.markdown(
        f"<div style='font-size:10px; overflow-y: hidden'>{content}</div>",
        unsafe_allow_html=True
    )

log_content.title("Logs")
main_content.title("Agentic AI Framework - LangGraph")

# Setup API clients
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="mock-local-key",
    # timeout=180.0,
    # max_retries=2
)

# Define state structure
class State(TypedDict):
    email_results: list
    calendar_results: list
    insights: str
    summary_feedback: str
    need_summary: bool
    iteration: int
    email: str
    event: str

class EmailGathererAgent:
    def gather_emails(self, state: State, max_retries: int = 3):
        time_frame = state["email"]
        util_st_log(f"Gathering emails for {time_frame}...")

        searched_emails_docs = vector_store.similarity_search(
            query = time_frame,
            k = 10 # top 3 most contecually relevent chunks
            )
        email_context = "\n".join([doc.page_content for doc in searched_emails_docs])
        return {"email_results": [email_context], "calendar_results":[], "iteration": 0, "insights": "", "summary_feedback": None, "email": state["email"], "event": state["event"]}

class EventGathererAgent:
    def gather_events(self, state: State, max_retries: int = 3):
        raw_query = state["event"]
        util_st_log(f"Gathering calendar events for {raw_query}...")
        time_min, time_max = parse_natural_language_date(raw_query)
        
        util_st_log(f"Gathering calendar events starting from {time_min or 'Now'}...")
        
        # 2. Pass the correctly formatted strings into your function
        calendar_results = get_calendar(time_min=time_min, time_max=time_max, max_results=20)
        return {"email_results": state["email_results"], "calendar_results":calendar_results, "iteration": 0, "insights": "", "summary_feedback": None, "email": state["email"], "event": state["event"]}


class AnalysisAgent:
    def analyze_data(self, state: State):
        util_st_log("AnalysisAgent started...")
        email_data = state ["email_results"]
        # raw_emails = str(state["email_results"])
        # truncated_emails = raw_emails[:4000] 
        
        # 2. Explicitly tell the model NOT to waste tokens on a thinking process
        prompt = (
            "You are an expert data analyst.\n"
            "Summarize the following higly relevant email and/or calendar data in clear, informative built points, highlighting in chronological order and most important/urgent emails and/or events. For each section have an emoji alongside heading.\n"
            "CRITICAL: Do not write a thinking process, intro notes, or internal monologue. "
            "Output ONLY the final summary paragraphs directly.\n\n"
            f"Data\n"
            f" - Calendar events:{str(state['calendar_results'])}"
            f" - Emails snippets:{email_data}\n"
        )
        if state.get("summary_feedback"):
            prompt += f"\n\nFeedback from reviewer to incorporate: {state['summary_feedback']}"

        try:
            response = client.chat.completions.create(
                # model="mlx-community/gemma-4-e4b-it-4bit",

                model = "jedisct1/gemma-4-E2B-it-txt-mlx-4bit",
                # STRICTLY use only 'user' role for Gemma
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000
            )

            message = response.choices[0].message
            
            if hasattr(message, 'content') and message.content:
                insights = message.content.strip()
            elif hasattr(message, 'reasoning') and message.reasoning:
                # Local server put the actual text inside the reasoning block
                insights = message.reasoning.strip()
                # Clean up internal "thinking process" meta-text if the model ignored our prompt
                if "thinking process" in insights.lower():
                    util_st_log("Note: Model returned thinking process as insights. Using it as fallback.")
            else:
                insights = "Error: The model returned a response, but both content and reasoning fields were empty."
        
        except Exception as e:
            insights = f"Analyze agent API Error ({type(e).__name__}): {str(e)}"
            print("\n\n ",insights)
        util_st_log(f"Insights generated ({len(insights)} chars): {insights[:50]}...")
        return {"email_results": state["email_results"],  "calendar_results":state["calendar_results"], "insights": insights, "iteration": state["iteration"], "summary_feedback": None, "email": state["email"], "event": state["event"]}

class ReviewerAgent:
    def review_insights(self, state: State, max_iterations=2):
        time.sleep(2)
        review_prompt = f"You are an expert reviewer.\n\nReview the following email and/or calendar event report for clarity and quality. If revision is needed, start your response with exactly 'Needs revision'.\n\nReport:\n{state['insights']}"
        
        try:
            response = client.chat.completions.create(
                # model="mlx-community/gemma-4-e4b-it-4bit",
                model = "jedisct1/gemma-4-E2B-it-txt-mlx-4bit",
                # STRICTLY use only 'user' role for Gemma
                messages=[{"role": "user", "content": review_prompt}],
                max_tokens=2000
            )

            # Safety fallback to prevent NoneType errors
            message = response.choices[0].message
            if hasattr(message, 'content') and message.content:
                summary_feedback = message.content.strip()
            elif hasattr(message, 'reasoning') and message.reasoning:
                summary_feedback = message.reasoning.strip()
            else:
                summary_feedback = ""
        
        except Exception as e:
            error_msg = f"Review agent API Error ({type(e).__name__}): {str(e)}"
            summary_feedback = error_msg
            util_st_log(f"<span style='color:red'>{error_msg}</span>")

        # This will no longer crash because summary_feedback is guaranteed to be a string
        need_revision = "Needs revision" in summary_feedback and state["iteration"] < max_iterations

        util_st_log(f"Review feedback: {summary_feedback[:50]}... | Needs revision: {need_revision}")
        
        return {"email_results": state["email_results"],  "calendar_results":state["calendar_results"], "insights": state["insights"], "iteration": state["iteration"] + 1, "summary_feedback": summary_feedback if need_revision else None, "email": state["email"], "event": state["event"], "need_summary": need_revision}

class ReportCompilerAgent:
    def compile_report(self, state: State):
        main_content.write(state["insights"])
        return state

# Instantiate agents
email_gatherer = EmailGathererAgent()
event_gatherer = EventGathererAgent()
analysis_agent = AnalysisAgent()
reviewer_agent = ReviewerAgent()
report_compiler = ReportCompilerAgent()

# Build the state graph
builder = StateGraph(State)
builder.add_node("gather_emails", email_gatherer.gather_emails)
builder.add_node("gather_events", event_gatherer.gather_events)
builder.add_node("analyze_data", analysis_agent.analyze_data)
builder.add_node("review_insights", reviewer_agent.review_insights)
builder.add_node("compile_report", report_compiler.compile_report)

builder.add_edge(START, "gather_emails")
builder.add_edge("gather_emails","gather_events")
builder.add_edge("gather_events", "analyze_data")

# FIXED: Changed from conditional_edges to standard add_edge since it's a guaranteed path
builder.add_edge("analyze_data", "review_insights") 

builder.add_conditional_edges("review_insights", lambda state: "analyze_data" if state["need_summary"] else "compile_report")

# FIXED: Added the final edge to tell LangGraph where the execution ends
builder.add_edge("compile_report", END)

graph = builder.compile()

# User input and process trigger
week = main_content.text_input("Enter the weeks of email summary", "last 2 week summary")
if main_content.button("Generate Report"):
    initial_state = {"email_results": [], "calendar_results":[], "insights": "", "iteration": 0, "summary_feedback": None, "email": week,"event":week, "need_summary": False}
    # Using streamlit spinner for better UX while running
    with st.spinner(f"Agents are researching {week}... Check logs for details."):
        graph.invoke(initial_state)