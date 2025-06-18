# diagnostic_agents.py
import asyncio
import nest_asyncio
from agents import Agent, Runner
from pydantic import BaseModel

nest_asyncio.apply()

class DifferentialDiagnosis(BaseModel):
    diagnoses: list[str]
    explanation: str

class ImageGuardrail(BaseModel):
    is_relevant: bool
    reason: str

image_guardrail_agent = Agent(
    name="Image Guardrail",
    instructions="""
        You are a filter that evaluates whether an uploaded image is relevant for medical or biological diagnostic purposes.
        You should return {"is_relevant": true, "reason": "..."} if the image looks appropriate for generating a differential diagnosis.
        Otherwise, return {"is_relevant": false, "reason": "..."}.
    """,
    output_type=ImageGuardrail,
    model="gpt-4o"
)

diagnostic_agent = Agent(
    name="Diagnostic Agent",
    instructions="""
        You are a medical assistant trained in dermatology, radiology, and general clinical diagnostics.
        Given a relevant image, you return a differential diagnosis in JSON format with a list of possible conditions and a brief explanation.
        You must never make a definitive diagnosis—only provide possibilities with reasoning.
        Use your visual understanding and base your answers only on the content of the image.
    """,
    model="gpt-4o",
    output_type=DifferentialDiagnosis,
    input_type="image"
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="""
        You are responsible for routing images to the diagnostic agent or rejecting them if irrelevant.
        First, consult the Image Guardrail. If relevant, pass to the Diagnostic Agent and return the result.
        If not relevant, return a message stating why.
    """,
    handoffs=[image_guardrail_agent, diagnostic_agent],
    model="gpt-4o"
)

def run_async_task(task):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(task)

def get_differential(image_path: str):
    with open(image_path, "rb") as img:
        result = run_async_task(Runner.run(triage_agent, img.read()))
    return result.final_output
