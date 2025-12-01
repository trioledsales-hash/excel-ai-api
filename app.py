from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import requests

# ========== CONFIG ==========

# Get OpenRouter API key from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set in environment variables.")

# Path to your Excel file (make sure this file is present on the server)
EXCEL_FILE_PATH = "Pricelistsheet.xlsx"

# OpenRouter endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Your chosen model on OpenRouter
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"


# ========== LOAD EXCEL ON STARTUP ==========

def load_excel_as_text(excel_path: str) -> str:
    """Read all sheets from the Excel file and flatten them into a text block."""
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


# ========== FASTAPI APP SETUP ==========

app = FastAPI(title="Excel AI API (OpenRouter)")

# CORS so your Hostinger site can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict to ["https://triosalesin.in"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== REQUEST MODEL ==========

class Question(BaseModel):
    question: str


# ========== CORE LOGIC ==========

def answer_from_excel(question: str) -> str:
    """Send Excel content + user question to OpenRouter and return answer."""

    system_prompt = """
You are an assistant that answers questions using ONLY the provided Excel content.
If the answer is not found in the Excel data, reply exactly: "I don't know based on the Excel data."
Do not invent any information.
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
        # Optional but good practice for OpenRouter analytics:
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
        # Try to expose OpenRouter's error message if available
        try:
            err_json = res.json()
            detail = err_json.get("error", {}).get("message", str(e))
        except Exception:
            detail = str(e)
        raise HTTPException(status_code=res.status_code, detail=f"OpenRouter error: {detail}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter request failed: {e}")

    json_res = res.json()

    try:
        return json_res["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid response format from OpenRouter")


# ========== ROUTES ==========

@app.post("/ask")
async def ask_q(payload: Question):
    try:
        answer = answer_from_excel(payload.question)
        return {"answer": answer}
    except HTTPException:
        # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        print("ERROR in /ask:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok", "message": "Excel AI API (OpenRouter) is running."}
