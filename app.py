from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import requests
import re

# ========================
#   CONFIG
# ========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set.")

EXCEL_FILE_PATH = "Pricelistsheet.xlsx"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Primary model: DeepSeek Chat (FREE)
PRIMARY_MODEL = "deepseek/deepseek-chat:free"

# Fallback model: Llama 3.3 70B FREE
FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


# ========================
#   LOAD EXCEL INTO DATAFRAMES
# ========================

if not os.path.exists(EXCEL_FILE_PATH):
    raise RuntimeError(f"Excel file not found: {EXCEL_FILE_PATH}")

xls = pd.ExcelFile(EXCEL_FILE_PATH)
SHEETS: dict[str, pd.DataFrame] = {}
for sheet_name in xls.sheet_names:
    SHEETS[sheet_name] = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet_name)


def get_relevant_excel_context(question: str, max_rows: int = 40) -> str:
    """
    Search Excel rows matching keywords in question.
    Returns only relevant rows (improves accuracy).
    """
    words = re.findall(r"\w+", question.lower())
    stopwords = {
        "what","is","the","price","of","and","all","in","on","to",
        "show","list","give","item","items","excel","sheet","from",
        "mrp","rate","for","brand","light","watt","w","with","street"
    }

    keywords = [w for w in words if w not in stopwords]

    if not keywords:
        return ""

    contexts = []
    per_sheet_limit = max(5, max_rows // max(len(SHEETS), 1))

    for sheet_name, df in SHEETS.items():
        df_low = df.astype(str).apply(lambda c: c.str.lower())
        mask = pd.Series(False, index=df.index)

        for kw in keywords:
            row_match = df_low.apply(lambda r: r.str.contains(kw, na=False)).any(axis=1)
            mask |= row_match

        matched = df[mask]
        if not matched.empty:
            matched = matched.head(per_sheet_limit)
            sheet_text = f"Sheet: {sheet_name}\n" + matched.to_string(index=False)
            contexts.append(sheet_text)

    return "\n\n".join(contexts)


# ========================
#   FASTAPI
# ========================

app = FastAPI(title="Excel AI API (DeepSeek version)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


# ========================
#   AI LOGIC
# ========================

def call_openrouter_model(model_name: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://triosalesin.in",
        "X-Title": "Trio Sales Excel AI",
    }

    data = {
        "model": model_name,
        "messages": prompt,
        "temperature": 0.0
    }

    res = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)

    if res.status_code == 429:
        raise Exception(f"{model_name} rate-limited (429). Try again later.")
    if res.status_code >= 400:
        try:
            detail = res.json().get("error", {}).get("message", res.text)
        except:
            detail = res.text
        raise Exception(f"{model_name} error: {detail}")

    try:
        return res.json()["choices"][0]["message"]["content"].strip()
    except:
        raise Exception(f"Invalid response format from {model_name}")


def answer_from_excel(question: str) -> str:
    context = get_relevant_excel_context(question)

    if not context.strip():
        return "I don't know based on the Excel data."

    system_prompt = """
You answer ONLY using the Excel rows provided.
Rules:
- Never guess. Never make up numbers.
- If the answer is not clearly found in the rows, reply exactly:
  "I don't know based on the Excel data."
- For product lists, ALWAYS return a Markdown table:

| Item | MRP | Platinum | Gold | Silver | Bronze |
|------|------|----------|--------|---------|---------|
| Example | 120 | 80 | 85 | 90 | 95 |

- Do NOT include explanations, reasoning, or extra text.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Excel rows:\n{context}\n\nQuestion: {question}"}
    ]

    # Try DeepSeek first
    try:
        return call_openrouter_model(PRIMARY_MODEL, messages)
    except Exception as deepseek_error:
        print("DeepSeek failed:", deepseek_error)

        # Try fallback Llama
        try:
            return call_openrouter_model(FALLBACK_MODEL, messages)
        except Exception as fallback_error:
            print("Fallback also failed:", fallback_error)
            raise HTTPException(
                status_code=502,
                detail=f"DeepSeek failed: {deepseek_error}; Fallback failed: {fallback_error}"
            )


# ========================
#   ROUTES
# ========================

@app.post("/ask")
async def ask(payload: Question):
    try:
        return {"answer": answer_from_excel(payload.question)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok", "message": "Excel AI API (DeepSeek version) is running"}
