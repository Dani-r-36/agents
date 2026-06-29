import os
import time
import random
import streamlit as st
from langgraph.graph import START, END, StateGraph
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from openai import OpenAI
from typing_extensions import TypedDict

# Configure Streamlit layout with two columns: main content and logs
st.set_page_config(layout="wide")
main_content, log_content = st.columns(2)

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
    api_key="mock-local-key"
)
ddgs = DDGS()

# Define state structure
class State(TypedDict):
    search_results: list
    insights: str
    review_feedback: str
    need_review: bool
    iteration: int
    airline: str

class DataGathererAgent:
    def gather_data(self, state: State, max_retries: int = 3):
        airline = state["airline"]
        util_st_log(f"Gathering data for {airline}...")
        queries = [
            f"{airline} airline history overview",
            f"{airline} airline leadership structure",
            f"{airline} airline fleet, routes, market presence",
            f"{airline} airline revenue, growth, market share",
            f"{airline} airline partnerships and expansi>on",
            f"{airline} airline recent news and updates"
        ]

        search_results = []
        for query in queries:
            for attempt in range(max_retries):
                try:
                    results = ddgs.text(query, max_results=1)
                    util_st_log(f"Search results for '{query}': {results}")
                    if results:
                        search_results.extend(results)
                    time.sleep(random.uniform(1.0, 3.0))
                    break
                except DDGSException as e:
                    util_st_log(f"Search error: {str(e)}")
                    if "Ratelimit" in str(e):
                        wait_time = (attempt + 1) * 5
                        util_st_log(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        util_st_log(f"Max retries reached for query: {query}. Continuing with available data.")

        if not search_results:
            util_st_log("No search results obtained. Using fallback data.")
            search_results = [{
                "title": f"About {airline}",
                "body": f"Fallback information about {airline}. This is placeholder data."
            }]
        return {"search_results": search_results, "iteration": 0, "insights": "", "review_feedback": None, "airline": state["airline"]}

class AnalysisAgent:
    def analyze_data(self, state: State):
        search_content = "\n".join([result["body"] for result in state["search_results"] if "body" in result])
        
        # FIXED: Actually appended the search_content to the prompt so the model has context
        prompt = f"Summarize the following search content about {state['airline']} in clear, informative paragraphs.\n\nData:\n{search_content}"
        
        if state.get("review_feedback"):
            prompt += f"\n\nFeedback from reviewer to incorporate: {state['review_feedback']}"

        response = client.chat.completions.create(
            model="mlx-community/gemma-4-e4b-it-4bit",
            messages=[{"role": "user", "content": f"You are an expert data analyst. {prompt}"}],
            max_tokens=1000
        )

        insights = response.choices[0].message.content
        if not insights:
                insights = "Error: The model generated an empty response. Check server logs."
        util_st_log(f"Insights generated ({len(insights)} chars): {insights[:50]}...")
        return {"search_results": state["search_results"], "insights": insights, "iteration": state["iteration"], "review_feedback": None, "airline": state["airline"]}

class ReviewerAgent:
    def review_insights(self, state: State, max_iterations=2):
        # FIXED: Combined the instructions and the actual insights into a single prompt for the user role
        review_prompt = f"Review the following airline report for clarity and quality. If revision is needed, start your response with exactly 'Needs revision'.\n\nReport:\n{state['insights']}"
        
        response = client.chat.completions.create(
            model="mlx-community/gemma-4-e4b-it-4bit",
            messages=[{"role": "system", "content": f"You are an expert reviewer.{review_prompt}"}] # FIXED: Passed the correct variable
        )

        review_feedback = response.choices[0].message.content
        need_revision = "Needs revision" in review_feedback and state["iteration"] < max_iterations

        util_st_log(f"Review feedback: {review_feedback[:50]}... | Needs revision: {need_revision}")
        
        return {"search_results": state["search_results"], "insights": state["insights"], "iteration": state["iteration"] + 1, "review_feedback": review_feedback if need_revision else None, "airline": state["airline"], "need_review": need_revision}

class ReportCompilerAgent:
    def compile_report(self, state: State):
        main_content.write(state["insights"])
        return state

# Instantiate agents
data_gatherer = DataGathererAgent()
analysis_agent = AnalysisAgent()
reviewer_agent = ReviewerAgent()
report_compiler = ReportCompilerAgent()

# Build the state graph
builder = StateGraph(State)
builder.add_node("gather_data", data_gatherer.gather_data)
builder.add_node("analyze_data", analysis_agent.analyze_data)
builder.add_node("review_insights", reviewer_agent.review_insights)
builder.add_node("compile_report", report_compiler.compile_report)

builder.add_edge(START, "gather_data")
builder.add_edge("gather_data", "analyze_data")

# FIXED: Changed from conditional_edges to standard add_edge since it's a guaranteed path
builder.add_edge("analyze_data", "review_insights") 

builder.add_conditional_edges("review_insights", lambda state: "analyze_data" if state["need_review"] else "compile_report")

# FIXED: Added the final edge to tell LangGraph where the execution ends
builder.add_edge("compile_report", END)

graph = builder.compile()

# User input and process trigger
airline = main_content.text_input("Enter the name of an airline", "Cathay Pacific")
if main_content.button("Generate Report"):
    initial_state = {"search_results": [], "insights": "", "iteration": 0, "review_feedback": None, "airline": airline, "need_review": False}
    # Using streamlit spinner for better UX while running
    with st.spinner(f"Agents are researching {airline}... Check logs for details."):
        graph.invoke(initial_state)