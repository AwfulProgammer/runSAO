"""
Agent Streams with Splunk AO — Sample Application

Demonstrates how to send traces to a Splunk AO Agent Stream using the
Splunk AO OpenAI wrapper. All LLM calls are automatically logged —
no manual instrumentation needed.

This script runs a few different prompts; open your Splunk Agent Observability UI
to see the traces appear in your Agent Stream.

This script assumes the .env file is located
in the parent directory, and it's properly configured.

Setup:
    1. review the .env configuration file
    2. pip install -r requirements.txt
    3. python 01_sample_app.py
"""

import os
from dotenv import load_dotenv
from splunk_ao.openai import openai
from splunk_ao import splunk_ao_context

load_dotenv("../.env")
project=os.environ.get("SPLUNK_AO_PROJECT")
agent_stream=os.environ.get("SPLUNK_AO_AGENT_STREAM")

# Confirm that environment variables are properly mapped
print(f"Loaded env variables: splunk ao realm: {os.environ.get("SPLUNK_AO_REALM")} | Project: {project} | Agent Stream: {agent_stream}")

# Splunk AO wraps the OpenAI client automatically.
# Traces are sent to the project and agent stream set in your .env file.
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def ask(question: str) -> str:
    """Send a question to the model and return the response."""
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    answer = response.choices[0].message.content
    print(f"Q: {question}")
    print(f"A: {answer}\n")
    return answer


if __name__ == "__main__":
    # Run a few sample prompts — each becomes a trace in your Agent Stream
    with splunk_ao_context(project=project, agent_stream=agent_stream):
        splunk_ao_context.start_session(name="Lab 01 - Why Agent Observability Matters")
        ask("Summarize the impact of rising interest rates on technology sector and how it might affect valuations and investor sentiment in the short term.")
        ask("What are the best growth stocks to invest in?")
        ask("What would be the impact of rising interest rages?")

    print("✓ Done — open your Splunk Agent Observability UI to see the traces.")
    print(f"  Project:    {os.environ.get('SPLUNK_AO_PROJECT')}")
    print(f"  Agent Stream: {os.environ.get('SPLUNK_AO_AGENT_STREAM')}")
