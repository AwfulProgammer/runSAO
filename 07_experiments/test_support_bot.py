import os
import time
from dotenv import load_dotenv
from splunk_ao.openai import openai
from splunk_ao import log
from splunk_ao.schema.metrics import SplunkAOEvaluators
from splunk_ao.experiments import run_experiment, get_experiment

load_dotenv("../.env")
project=os.environ.get("SPLUNK_AO_PROJECT")
agent_stream=os.environ.get("SPLUNK_AO_AGENT_STREAM")

# Confirm that environment variables are properly mapped
print(f"Loaded env variables: splunk ao realm: {os.environ.get("SPLUNK_AO_REALM")} | Project: {project} | Agent Stream: {agent_stream}")


client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── Mock knowledge base (simulates a vector store retrieval) ──────────────────

KNOWLEDGE_BASE = {
    "climate": (
        "Global average temperatures have risen approximately 1.2°C above pre-industrial levels. "
        "The IPCC's Sixth Assessment Report projects continued warming of 1.5–4.5°C by 2100, "
        "depending on emissions trajectories. The last decade (2011–2020) was the warmest on record."
    ),
    "ai": (
        "Large language models (LLMs) are trained on vast text corpora using self-supervised learning "
        "with transformer architectures. Emergent capabilities such as few-shot reasoning and code "
        "generation appear at scale. RLHF (Reinforcement Learning from Human Feedback) is commonly "
        "used to align LLMs with human preferences."
    ),
    "NASA": (
        "NASA's Artemis program is designed to return humans to the lunar surface. Artemis III, "
        "targeting the lunar south pole, uses SpaceX's Starship as the human landing system. "
        "The program aims to establish a sustained presence on and around the Moon by the late 2020s."
    ),
}

SYSTEM_PROMPT = (
    "You are a concise research assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer under 3 sentences. "
)

# ── Agent functions ────────────────────────────────────────────────────────────

@log(span_type="retriever")
def retrieve_context(question: str) -> str:
    """Retrieve the most relevant context chunk for the question."""
    question_lower = question.lower()
    for keyword, context in KNOWLEDGE_BASE.items():
        if keyword.lower() in question_lower:
            return context
    return ""


@log(span_type="llm")
def generate_answer(question: str, context: str) -> str:
    """Generate an answer grounded in the retrieved context."""
    user_message = (
        f"Context:\n{context}\n\nQuestion: {question}"
        if context
        else f"Question: {question}"
    )
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=200,
    )
    return response.choices[0].message.content.strip()


@log(span_type="agent")
def run_rag_agent(question: str) -> dict:
    """Full RAG pipeline: retrieve context then generate a grounded answer."""
    context = retrieve_context(question)
    answer = generate_answer(question, context)
    return {"question": question, "context": context, "answer": answer}


# ── Part 1: Experiment with preset metrics ─────────────────────────────────────

def test_support_bot_correctness():
    """
    Run an offline experiment against a fixed dataset, scoring each row
    with Splunk AO's out-of-the-box metrics.
    Scores appear in the Experiments tab in the Splunk Agent Observability UI.
    """

    # Dataset: list of (question, expected_answer) pairs
    dataset = [
        {
            "input": "What is happening with climate change?",
            "expected_output": "Global temperatures have risen ~1.2°C and continue to rise.",
        },
        {
            "input": "How do AI large language models work?",
            "expected_output": "LLMs use transformer architectures trained on large text corpora.",
        },
        {
            "input": "What is NASA's Artemis program?",
            "expected_output": "Artemis aims to return humans to the Moon using SpaceX Starship.",
        },
        # {
        #     "input": "Tell me about the history of jazz music.",  # out-of-domain — no context
        #     "expected_output": "Jazz originated in New Orleans in the early 20th century.",
        # }        
    ]


    print("\n[Experiment] Running offline evaluation with preset metrics...")
    results = run_experiment(
        experiment_name="ci-support-bot-test",
        dataset=dataset,
        function=run_rag_agent,
        metrics=[SplunkAOEvaluators.completeness],
        project=project
    )

    exp_response = results['experiment'].to_dict()
    experiment_name = exp_response['name']

    while True:
        experiment_result = get_experiment(project_name = project, experiment_name=experiment_name)
        experiment = experiment_result.to_dict()
        metrics = experiment['aggregate_metrics']
        try:
            avg = metrics['average_completeness_gpt']
            if avg >= 0.80:
                print(f"Test result successful with {avg:.2f} for a required 0.80 threshold. Ready for merging.")
                break
            else:
                print(f"Correctness {avg:.2f} is below the required 0.80 threshold. Review the experiment before merging.")
                break
        except Exception as ex:
            print("Metrics not yet computed. Trying again in 5 seconds")
            time.sleep(5)   


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Splunk AO Experiment Test")
    print("=" * 60)

    # Offline experiment with explicit metric selection
    test_support_bot_correctness()

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Open the Splunk Agent Observability UI → Experiments and find this experiment.")
    print("  2. Navigate throught the traces to see the metrics scores.")