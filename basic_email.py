import json
from rich.markdown import Markdown
import datetime
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


from bs4 import BeautifulSoup
import datetime 
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# If modifying these scopes, delete the file token.json.
SCOPES_CALDENDAR = ["https://www.googleapis.com/auth/calendar.readonly"]

def get_service(email_calendar):
    """
    A function that provides creds for Gmail

    """
    creds = None
    token_name = "email_token.json" if email_calendar else "calendar_token.json"
    scopes = SCOPES if email_calendar else SCOPES_CALDENDAR
    if os.path.exists(token_name):
        creds = Credentials.from_authorized_user_file(token_name, scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('creds.json', scopes)
            creds = flow.run_local_server(port=0)
        with open(token_name, 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds) if email_calendar else build("calendar", "v3", credentials=creds)

# def get_calendar():
#     """
#     A function that returns last 10 calendar events
#     """
#     service = get_service(email_calendar=False)
#     # Call the Calendar API
#     now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
#     print("Getting the upcoming 10 events")
#     events_result = (
#         service.events()
#         .list(
#             calendarId="primary",
#             timeMin=now,
#             maxResults=10,
#             singleEvents=True,
#             orderBy="startTime",
#         )
#         .execute()
#     )
#     events = events_result.get("items", [])

#     if not events:
#         print("No upcoming events found.")
#         return
#     return events

def get_calendar(time_min: str = None, time_max: str = None, max_results: int = 10):
    """
    Returns calendar events within a specified timeframe.
    
    Args:
        time_min: The start datetime in ISO 8601 format. Defaults to current time if None.
        time_max: The end datetime in ISO 8601 format. Optional.
        max_results: The maximum number of events to fetch.
    """
    service = get_service(email_calendar=False)
    
    if time_min is None:
        time_min = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    
    # 2. DEFENSIVE FIX: Ensure time_min has a timezone indicator
    if time_min and not time_min.endswith('Z') and '+' not in time_min[-6:] and '-' not in time_min[-6:]:
        time_min += 'Z'
        
    # 3. DEFENSIVE FIX: Ensure time_max has a timezone indicator (if provided)
    if time_max and not time_max.endswith('Z') and '+' not in time_max[-6:] and '-' not in time_max[-6:]:
        time_max += 'Z'  
    print(f"Getting up to {max_results} events starting from {time_min}")
    
    # Build the base arguments for the API call
    api_args = {
        "calendarId": "primary",
        "timeMin": time_min,
        "maxResults": max_results,
        "singleEvents": True,
        "orderBy": "startTime",
    }
    
    # Add timeMax only if the LLM provided it
    if time_max:
        api_args["timeMax"] = time_max

    # Execute the API call using the unpacked dictionary
    events_result = service.events().list(**api_args).execute()
    events = events_result.get("items", [])

    if not events:
        print("No events found for this timeframe.")
        return "No events found for the requested timeframe."
    print(events)
    return events


def get_emails():
    """
    A function that returns emails from last 2 weeks

    """
    service = get_service(email_calendar=True)
    two_weeks_ago = (datetime.datetime.now() - datetime.timedelta(days=14)).strftime('%Y/%m/%d')
    query = f"after:{two_weeks_ago}"
    results = service.users().messages().list(maxResults=100,q=query, userId='me').execute()

    # We can also pass maxResults to get any number of emails. Like this:
    # result = service.users().messages().list(maxResults=200, userId='me').execute()
    messages = results.get('messages')

    # messages is a list of dictionaries where each dictionary contains a message id.

    if not messages:
            return "You have no unread emails."
        
    summary = []
    for msg in messages:
        m = service.users().messages().get(userId='me', id=msg['id']).execute()
        ms_timestamp = int(m['internalDate'])
        date_obj = datetime.datetime.fromtimestamp(ms_timestamp / 1000.0)
        readable_date = date_obj.strftime('%b %d, %Y %H:%M')
        headers = m['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "(No Subject)")
        sender = next((h['value'] for h in headers if h['name'] == 'From'), "(Unknown Sender)")
        summary.append(f"- From: {sender} | Subject: {subject} | Date: {readable_date}")
            
    return "\n".join(summary)

# ---- Config ----
SUMMARY_UPDATE_EVERY = 3  # update summary every N turns

MAX_ITERATIONS = 3
CURRENT_ITERATION = 0
MODEL_NAME = "mlx-community/gemma-4-e4b-it-4bit"
last_fetch_time = None
model, tokenizer = load(MODEL_NAME)
console = Console()
# ---- Memory ----
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

summary_text = ""  # long-term memory
turn_count = 0

def get_current_date_time():
    """Returns the current local time."""
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def build_system_prompt(summary_text, is_stale):
    date_now = get_current_date_time()
    if is_stale:
        return (
            f"You are a helpful personal assistant. Current Date Time: {date_now}\n"
            f"STATUS: STALE. You have NO email data in memory.\n"
            f"MANDATORY: You MUST call 'get_emails()' before answering any questions about emails if STATUS is STALE."
            f"SUMMARY: This is the previous chats summary, if empty then no previous summary available :{summary_text}."
        )
    else:
        return (
            f"You are a helpful personal assistant. Current Date Time: {date_now}\n"
            f"STATUS: FRESH. The current email data is provided below in the responses where the 'role' is 'tool' and the 'name' is 'get_emails',.\n"
            f"TASK: Use the provided email list and calendar events to answer the user's request accurately."
            f"SUMMARY: This is the previous chats summary, if empty then no previous summary available :{summary_text}."
        )


def update_summary(model, tokenizer, messages, old_summary):
    """Generate a compressed memory of the conversation."""
    
    summary_messages = [
        {
            "role": "system",
            "content": (
                "Summarize the conversation into a short memory. "
                "Keep key facts, user preferences, and important results."
            )
        },
        {
            "role": "user",
            "content": f"Previous summary:\n{old_summary}\n\nNew messages:\n{messages[-6:]}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        summary_messages, add_generation_prompt=True
    )

    new_summary = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=300,
        sampler=make_sampler(temp=0.0),
    ).strip()

    return new_summary

# ---- Main loop ----
def chat(user_input):
    global messages, summary_text, turn_count, MAX_ITERATIONS, last_fetch_time
    turn_count += 1
    current_iteration = 0 # Local reset for each user message
    now = datetime.datetime.now()
    is_stale = True
    tools ={
        "get_emails": get_emails,
        "get_calendar": get_calendar,
        "get_current_date_time": get_current_date_time
    }
    if last_fetch_time:
        is_stale = (now - last_fetch_time).total_seconds() > 120

    # --- NEW: Context Pruning ---
    # If stale, we remove old tool results so the model CANNOT cheat
    if is_stale:
        # Keep the system prompt (index 0) but filter out previous tool results
        print("data is stale about to delete")
        messages = [messages[0]] + [m for m in messages[1:] if m.get("role") != "tool" and "TOOL_RESULT" not in m.get("content", "")]
        print(messages)

    # 1. Update System Prompt with the status
    prompt_build = build_system_prompt(summary_text, is_stale)
    print(prompt_build)
    messages[0] = {"role": "system", "content": prompt_build}
    messages.append({"role": "user", "content": user_input})

    while current_iteration < MAX_ITERATIONS:
        print(f"current it {current_iteration}")
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tools=list(tools.values()))
        # response = generate(model, tokenizer, prompt=prompt, max_tokens=1000, sampler=make_sampler(temp=0.0))
        # has_tool_tokens = (tokenizer.tool_call_start in response and tokenizer.tool_call_end in response)
        stream = stream_generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=1000, 
            # temperature=0.0  # Note: mlx_lm uses 'temp' or 'temperature' depending on version
        )

        response_text = ""
        is_tool_call_detected = False

        for response in stream:
            # Handle version compatibility: object with .text attribute vs raw string
            token_text = getattr(response, "text", str(response))
            
            response_text += token_text
            
            # Check if the model has started generating a tool call
            if tokenizer.tool_call_start in response_text:
                is_tool_call_detected = True
                continue  # Do NOT yield tool tokens to the user UI
            
            # If it's a normal conversation response, stream it out!
            if not is_tool_call_detected:
                yield token_text

        # --- AFTER STREAM FINISHES ---
        # Check if the fully accumulated response contains a tool call
        has_tool_tokens = (tokenizer.tool_call_start in response_text and tokenizer.tool_call_end in response_text)
        if has_tool_tokens:

            start_tool = response_text.find(tokenizer.tool_call_start) + len(tokenizer.tool_call_start)
            end_tool = response_text.find(tokenizer.tool_call_end)
            tool_call = tokenizer.tool_parser(response_text[start_tool:end_tool].strip())
            tool_result = tools[tool_call["name"]](**tool_call["arguments"])
            # print(f"this is tool_result\n{tool_result}")
        # if "TOOL_CALL: get_emails()" in response:
        #     print("System: Tool Triggered (Data was stale).")
            if tool_call["name"] == "get_emails":
                last_fetch_time = datetime.datetime.now()
            print("email tool called")
            # email_data = get_emails()
            
            # Save the clean call
            # idx = response.find("TOOL_CALL: get_emails()")
            # clean_call = response[:idx + len("TOOL_CALL: get_emails()")]
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"{tool_call["name"]}:\n{str(tool_result)}"})

            # messages.append({"role": "tool", "name": tool_call["name"], "content": str(tool_result)})
            new_system_prompt = build_system_prompt(summary_text, is_stale=False)
            messages[0] = {"role": "system", "content": new_system_prompt}
            current_iteration += 1
            print("updated messages")
            continue # Go back to start of loop to see if it needs another tool
        else:
            # No more tool calls; save the final text and break loop
            print("else hit")
            messages.append({"role": "assistant", "content": response_text})
            break

    # 4. Periodic Summary Update
    if turn_count % SUMMARY_UPDATE_EVERY == 0:
        print("System: Updating summary...")
        summary_text = update_summary(model, tokenizer, messages, summary_text)
        # Keep system prompt + last 10 messages to save context space
        messages = [messages[0]] + messages[-10:]

if __name__ == "__main__":
    question = None
    while True:
        question = input("Enter question to ask\nQ to quit\n")
        if question in ["Q", "q"]:
            break
        for chunk in chat(question):
            print(chunk, end="", flush=True)