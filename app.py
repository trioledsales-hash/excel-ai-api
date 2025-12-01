from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import requests

# ========================
#   CONFIG
# ========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set.")

EXCEL_FILE_PATH = "Pricelistsheet.xlsx"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Use Gemini Flash
OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"


# ========================
#   LOAD EXCEL
# ========================

def load_excel_as_text(excel_path: str) -> str:
    if not os.path.exists(excel_path):
        raise RuntimeError(f"Excel file not found: {excel_path}")

    xls = pd.ExcelFile(excel_path)
    sheet_texts = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        sheet_text = f"Sheet: {sheet_name}\n"
        sheet_text += df.to_string(index=False)
        sheet_texts.append(sheet_text)

    return "\n\n".join(sheet_texts)


EXCEL_TEXT = load_excel_as_text(EXCEL_FILE_PATH)


# ========================
#   FASTAPI SETUP
# ========================

app = FastAPI(title="Excel AI API (Gemini Flash)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can limit to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================
#   REQUEST MODEL
# ========================

class Question(BaseModel):
    question: str


# ========================
#   AI LOGIC
# ========================

def answer_from_excel(question: str) -> str:
    system_prompt = """
You are an assistant that answers ONLY using the Excel content provided.
When listing product data, ALWAYS return results in **Markdown table format** like this:

| Item | MRP | Platinum | Gold | Silver | Bronze |
|------|------|----------|--------|---------|---------|
| Example | 120 | 80 | 85 | 90 | 95 |

If the information is not present in Excel data, reply exactly:
"I don't know based on the Excel data."

Never hallucinate or guess.
"""

    user_content = f"""
EXCEL CONTENT:
{EXCEL_TEXT}

USER QUESTION:
{question}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://triosalesin.in",
        "X-Title": "Trio Sales Excel AI",
    }

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
    }

    try:
        res = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)
        res.raise_for_status()
    except requests.exceptions.HTTPError as e:
        try:
            detail = res.json().get("error", {}).get("message", str(e))
        except:
            detail = str(e)
        raise HTTPException(status_code=res.status_code, detail=f"OpenRouter error: {detail}")

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter request failed: {e}")

    try:
        msg = res.json()["choices"][0]["message"]["content"].strip()
        return msg
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid response format from OpenRouter")


# ========================
#   ROUTES
# ========================

@app.post("/ask")
async def ask_q(payload: Question):
    try:
        return {"answer": answer_from_excel(payload.question)}
    except HTTPException:
        raise
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok", "message": "Excel AI API using Gemini Flash is running"}
