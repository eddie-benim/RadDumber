import asyncio
import nest_asyncio
from agents import Agent, Runner
from pydantic import BaseModel
from agents import set_default_openai_key

nest_asyncio.apply()
set_default_openai_key(st.secrets["OPENAI_API_KEY"])

class DifferentialDiagnosis(BaseModel):
    diagnoses: list[str]
    explanation: str

class ImageGuardrail(BaseModel):
    is_relevant: bool
    reason: str

image_guardrail_agent = Agent(
    name="Image Guardrail",
    instructions="""
        You are a medical content filter. Your job is to determine whether an uploaded image is relevant for diagnostic radiology.
        Accept only medical images such as X-rays, CT scans, MRIs, or ultrasound images. Reject non-diagnostic or unrelated content.
        Respond in this format: {"is_relevant": true/false, "reason": "..."}
    """,
    output_type=ImageGuardrail,
    model="gpt-4o"
)

diagnostic_agent = Agent(
    name="Radiology Diagnostic Agent",
    instructions="""
        You are a radiology assistant trained to interpret diagnostic images like chest X-rays, CT scans, or MRI scans.
        Given an appropriate diagnostic image, you will return a differential diagnosis as a list of possible conditions with a brief explanation.
        You are not allowed to make a definitive diagnosis.
        Focus strictly on image content, without assuming patient history or clinical symptoms.
        Output format should follow this schema: {"diagnoses": [...], "explanation": "..."}
    """,
    model="gpt-4o",
    output_type=DifferentialDiagnosis
)

triage_agent = Agent(
    name="Radiology Triage Agent",
    instructions="""
        You are responsible for routing uploaded images through the proper pipeline.
        First, consult the Image Guardrail to determine if the input is relevant to diagnostic radiology.
        If the image is acceptable, forward it to the Radiology Diagnostic Agent.
        If not, return the reason why the image was rejected.
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
