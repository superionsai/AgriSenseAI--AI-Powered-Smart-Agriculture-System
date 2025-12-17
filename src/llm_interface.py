# src/llm_interface.py
import subprocess
import json
import re

DEFAULT_MODEL = "llama3"
DEFAULT_TIMEOUT = 600  # seconds


def _run_ollama(prompt, model=DEFAULT_MODEL, timeout=DEFAULT_TIMEOUT):
    """
    Run Ollama safely. Returns raw text output.
    """
    try:
        p = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        out, err = p.communicate(prompt, timeout=timeout)

        if err and not out:
            return f"Ollama error: {err.strip()}"

        return out.strip()

    except subprocess.TimeoutExpired:
        return "Ollama timeout: model took too long to respond."

    except FileNotFoundError:
        return "Ollama not found. Is ollama installed and in PATH?"


def parse_user_query(text, model=DEFAULT_MODEL):
    """
    Convert a farmer's natural language query into structured intent JSON.
    """
    prompt = f"""
Convert the following farmer query into STRICT JSON.

Keys:
- intent
- target_date
- variables
- location
- notes

Rules:
- Return ONLY JSON
- Use null if unknown

Query:
{text}
"""

    raw = _run_ollama(prompt, model=model)

    try:
        m = re.search(r"\{{.*\}}", raw, re.DOTALL)
        return json.loads(m.group(0))
    except Exception:
        # Safe fallback
        return {
            "intent": None,
            "target_date": None,
            "variables": None,
            "location": None,
            "notes": text,
        }


def generate_advisory_text(clean_advisory, model="llama3"):
    prompt = f"""
You are an agricultural extension officer.

Convert the structured advisory below into SIMPLE, CLEAR, FARMER-FRIENDLY text.

Rules:
- Use short sentences
- Avoid technical terms
- Do NOT add new advice
- Do NOT invent risks
- If no action is needed, clearly say so
- Output plain text (NOT JSON)

Advisory:
{json.dumps(clean_advisory, indent=2)}
"""
    return _run_ollama(prompt, model=model)

def translate_text(text, target_lang="hi", model="llama3"):
    lang_map = {
        "hi": "Hindi",
        "te": "Telugu",
        "ta": "Tamil",
        "kn": "Kannada",
        "mr": "Marathi"
    }

    lang_name = lang_map.get(target_lang, "Hindi")

    prompt = f"""
Translate the following agricultural advisory into SIMPLE {lang_name}.

Rules:
- Use farmer-friendly language
- Short sentences
- Do NOT add new advice
- Do NOT remove information
- Use common rural words

Text:
{text}
"""

    return _run_ollama(prompt, model=model)

