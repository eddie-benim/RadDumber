import asyncio
import nest_asyncio
import base64
from openai import OpenAI
from pydantic import BaseModel
from agents import Agent, Runner

nest_asyncio.apply()
client = OpenAI()

class DiagnosisItem(BaseModel):
    condition: str
    probability: int | None = None  # Now optional

class DifferentialDiagnosis(BaseModel):
    diagnoses: list[DiagnosisItem]
    explanation: str

postprocess_agent = Agent(
    name="Radiology Postprocessor",
    instructions="""
        You are a clinical assistant that receives a freeform diagnosis paragraph and converts it into structured JSON.
        Your output should look like this:
        {
            "diagnoses": [
                {"condition": "Cardiomegaly", "probability": 70},
                {"condition": "Pleural effusion", "probability": 30}
            ],
            "explanation": "Key features observed and their interpretation."
        }

        If the original paragraph does not contain explicit percentages, estimate them based on relative confidence.
        If confidence is too ambiguous, include diagnoses without percentages, or distribute them evenly.
        If nothing is relevant, return an empty diagnosis list and a brief explanation.
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
                    {
                        "type": "text",
                        "text": (
                            "You are a radiologist trained to interpret chest X-rays. "
                            "Analyze this image and compare it to commonly seen clinical cases. "
                            "Base your differential diagnosis on visual features such as consolidation, hyperlucency, cardiomegaly, pleural thickening, and costophrenic angle blunting. "
                            "Then estimate probability percentages based on how closely this image resembles confirmed examples you've seen in training. "
                            "Focus on conditions like pneumonia, CHF, pneumothorax, tumors, and tuberculosis. "
                            "Avoid uniform distributions unless truly appropriate."
                        )
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }
        ]
    )

    diagnosis_text = vision_response.choices[0].message.content
    result = run_async_task(Runner.run(postprocess_agent, diagnosis_text))
    return result.final_output
