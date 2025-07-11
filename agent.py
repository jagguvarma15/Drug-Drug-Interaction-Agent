from judgeval.tracer import Tracer, wrap
from openai import OpenAI

client = wrap(OpenAI())  # tracks all LLM calls
judgment = Tracer(project_name="my_project")

@judgment.observe(span_type="tool")
def format_question(question: str) -> str:
    # dummy tool
    return f"Question : {question}"

@judgment.observe(span_type="function")
def run_agent(prompt: str) -> str:
    task = format_question(prompt)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": task}]
    )
    return response.choices[0].message.content
    
run_agent("Explain me about Continous Monitoring for AI Agents in 150 words.")