import asyncio
import nest_asyncio
import openai
import base64
from agents import Agent, Runner
from pydantic import BaseModel

nest_asyncio.apply()

# Step 1: Post-processing structure (text-to-structured output)
class DifferentialDiagnosis(BaseModel):
    diagnoses: list[str]
    explanation: str

# Step 2: Post-processing agent (takes raw diagnosis text, parses it, adds structure)
postprocess_agent = Agent(
    name="Radiology Postprocessor",
    instructions="""
        You are a clinical assistant that takes in unstructured text describing a differential diagnosis from a medical image
        and converts it into a structured JSON object with this schema:
        {
            "diagnoses": ["Diagnosis A", "Diagnosis B", ...],
            "explanation": "Explanation of findings"
        }
        If the input is irrelevant or nonsensical, return diagnoses: [] and a short explanation.
    """,
    model="gpt-4o",
    output_type=DifferentialDiagnosis
)

# Step 3: Utility for async agent call
def run_async_task(task):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(task)

# Step 4: Primary function — vision model + agent postprocessing
def get_differential(image_bytes: bytes):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Ask gpt-4o vision to interpret image directly
    vision_response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a radiology assistant. Analyze this chest X-ray and describe the differential diagnosis."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }
        ]
    )

    diagnosis_text = vision_response["choices"][0]["message"]["content"]

    # Postprocess the diagnosis text through the agent
    result = run_async_task(Runner.run(postprocess_agent, diagnosis_text))
    return result.final_output
