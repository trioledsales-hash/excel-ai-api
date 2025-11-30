from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
import os

# ========== CONFIG ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXCEL_FILE_PATH = "Pricelistsheet.xlsx"   # your Excel file
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Excel AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== LOAD EXCEL ==========
def load_excel_as_text(excel_path: str) -> str:
    # Read all sheets
    xls = pd.ExcelFile(excel_path)
    sheet_texts = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        # Convert each sheet to readable text
        sheet_text = f"Sheet: {sheet_name}\n"
        sheet_text += df.to_string(index=False)
        sheet_texts.append(sheet_text)

    return "\n\n".join(sheet_texts)


EXCEL_TEXT = load_excel_as_text(EXCEL_FILE_PATH)

# ========== REQUEST MODEL ==========
class Question(BaseModel):
    question: str

# ========== ANSWER FUNCTION ==========
def answer_from_excel(question: str) -> str:
    system_prompt = """
You are an assistant that answers questions using ONLY the provided Excel content.
If the answer is not found in the Excel data, reply: "I don't know based on the Excel data."
Do not invent any information.
"""

    user_content = f"""
EXCEL CONTENT:
{EXCEL_TEXT}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-5-mini",  # or another available model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()

# ========== ROUTES ==========
@app.post("/ask")
async def ask_q(payload: Question):
    try:
        answer = answer_from_excel(payload.question)
        return {"answer": answer}
    except Exception as e:
        print("ERROR in /ask:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok", "message": "Excel AI API is running."}
