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

# Make sure this filename matches your uploaded Excel on Render
EXCEL_FILE_PATH = "Pricelistsheet.xlsx"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Primary model (Gemini Flash free)
OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"

# Fallback model (free Llama) if Gemini provider fails
FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


# ========================
#   LOAD EXCEL -> DATAFRAMES
# ========================

if not os.path.exists(EXCEL_FILE_PATH):
    raise RuntimeError(f"Excel file not found: {EXCEL_FILE_PATH}")

_xls = pd.ExcelFile(EXCEL_FILE_PATH)
SHEETS: dict[str, pd.DataFrame] = {}
for sheet_name in _xls.sheet_names:
    SHEETS[sheet_name] = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet_name)


def get_relevant_excel_context(question: str, max_rows: int = 40) -> str:
    """
    Search all sheets for rows that match keywords in the question.
    Returns a text block with only relevant rows per sheet.
    """

    # Extract simple keywords from the question
    words = re.findall(r"\w+", question.lower())
    stopwords = {
        "what", "is", "the", "price", "of", "for", "all", "and",
        "in", "on", "to", "show", "list", "give", "items", "item",
        "mrp", "rate", "sheet", "excel", "from", "with", "brand",
        "watt", "w", "light", "street", "products", "product"
    }
    keywords = [w for w in words if w not in stopwords]

    if not keywords:
        return ""

    contexts: list[str] = []
    per_sheet_limit = max(5, max_rows // max(len(SHEETS), 1))

    for sheet_name, df in SHEETS.items():
        # Convert all values to lower-case strings for searching
        df_str = df.astype(str).apply(lambda col: col.str.lower())
        mask = pd.Series(False, index=df.index)

        for kw in keywords:
            row_has_kw = df_str.apply(lambda row: row.str.contains(kw, na=False)).any(axis=1)
            mask = mask | row_has_kw

        matched = df[mask]
        if not matched.empty:
            matched = matched.head(per_sheet_limit)
            sheet_text = f"Sheet: {sheet_name}\n"
            sheet_text += matched.to_string(index=False)
            contexts.append(sheet_text)

    return "\n\n".join(contexts)


# ========================
#   FASTAPI SETUP
# ========================

app = FastAPI(title="Excel AI API (Gemini + Fallback)")

# Allow your website (and others) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict to ["https://triosalesin.in"]
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
    """
    1) Find relevant rows in Excel
    2) Send only those rows + question to OpenRouter
    3) Use Gemini Flash as primary, Llama as fallback
    """

    relevant_context = get_relevant_excel_context(question, max_rows=40)

    # If nothing relevant found, don't guess
    if not relevant_context.strip():
        return "I don't know based on the Excel data."

    system_prompt = """
You are an assistant that answers questions ONLY using the provided Excel rows.

Rules:
- Use ONLY the rows given in the context. Do NOT invent or guess any numbers or items.
- When listing products, ALWAYS answer using a Markdown table, like:

| Item | MRP | Platinum | Gold | Silver | Bronze |
|------|------|----------|--------|---------|---------|
| Example | 120 | 80 | 85 | 90 | 95 |

- Keep the answer focused on the question (filter to matching brand/watt/category).
- If the requested item/brand/watt is NOT present in the given rows,
  reply exactly: "I don't know based on the Excel data."
- Do NOT explain your reasoning. Just give a short direct answer and/or table.
"""

    user_content = f"""
RELEVANT EXCEL ROWS:
{relevant_context}

USER QUESTION:
{question}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://triosalesin.in",
        "X-Title": "Trio Sales Excel AI",
    }

    def call_model(model_name: str) -> str:
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,  # more deterministic / accurate
        }

        res = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)

        if res.status_code >= 400:
            # Provider or model error – raise a normal Exception so we can try fallback
            try:
                err_json = res.json()
                detail = err_json.get("error", {}).get("message", res.text)
            except Exception:
                detail = res.text
            raise Exception(f"{model_name} error: {detail}")

        try:
            json_res = res.json()
            return json_res["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise Exception(f"Invalid response from {model_name}: {e}")

    # Try primary (Gemini) first
    try:
        return call_model(OPENROUTER_MODEL)
    except Exception as primary_err:
        print("Primary model failed:", primary_err)
        # Try fallback (Llama)
        try:
            return call_model(FALLBACK_MODEL)
        except Exception as fallback_err:
            print("Fallback model also failed:", fallback_err)
            raise HTTPException(
                status_code=502,
                detail=f"Primary model failed: {primary_err}; Fallback failed: {fallback_err}",
            )


# ========================
#   ROUTES
# ========================

@app.post("/ask")
async def ask_q(payload: Question):
    try:
        answer = answer_from_excel(payload.question)
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        print("ERROR in /ask:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok", "message": "Excel AI API (Gemini + Fallback) is running"}
