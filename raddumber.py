import asyncio
import nest_asyncio
import base64
from openai import OpenAI
from pydantic import BaseModel
from agents import Agent, Runner

nest_asyncio.apply()
client = OpenAI()

# Output schema for structured diagnosis with probabilities
class DiagnosisItem(BaseModel):
    condition: str
    probability: int  # percentage

class DifferentialDiagnosis(BaseModel):
    diagnoses: list[DiagnosisItem]
    explanation: str

# Postprocessing agent
postprocess_agent = Agent(
    name="Radiology Postprocessor",
    instructions="""
        You are a clinical assistant that takes in unstructured text describing a differential diagnosis from a medical image
        and converts it into a structured JSON object with this schema:
        {
            "diagnoses": [
                {"condition": "Pneumonia", "probability": 65},
                {"condition": "Pulmonary edema", "probability": 25},
                {"condition": "Pleural effusion", "probability": 10}
            ],
            "explanation": "Explanation of findings"
        }

        All diagnoses must include estimated probabilities (that add up to roughly 100%).
        Format the diagnosis names as concise conditions.
        If the input is irrelevant or nonsensical, return an empty diagnosis list and a brief explanation.
    """,
    model="gpt-4o",
    output_type=DifferentialDiagnosis
)

def run_async_task(task):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(task)

def get_differential(image_bytes: bytes):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    vision_response = client.chat.completions.create(
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

    diagnosis_text = vision_response.choices[0].message.content
    result = run_async_task(Runner.run(postprocess_agent, diagnosis_text))
    return result.final_output
