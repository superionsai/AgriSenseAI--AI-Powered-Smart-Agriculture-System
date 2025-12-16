# src/llm_interface.py
import subprocess
import json
import re

def _run_ollama(prompt, model="llama:7b", timeout=180):
    p = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate(prompt, timeout=timeout)
    return out.strip()

def parse_user_query(text, model="llama:7b"):
    prompt = f"""
Convert the following farmer query into STRICT JSON:

Keys:
- intent
- target_date
- variables
- location
- notes

Return ONLY JSON.

Query:
{text}
"""
    raw = _run_ollama(prompt, model=model)
    try:
        m = re.search(r"\{{.*\}}", raw, re.DOTALL)
        return json.loads(m.group(0))
    except Exception:
        return {
            "intent": None,
            "target_date": None,
            "variables": None,
            "location": None,
            "notes": text,
        }

def generate_advisory_text(data, model="llama:7b"):
    prompt = f"""
You are an agricultural expert.

From the JSON below, produce:
- summary
- risks
- recommendations
- actions

Return STRICT JSON.

Input:
{json.dumps(data, indent=2)}
"""
    raw = _run_ollama(prompt, model=model)
    try:
        m = re.search(r"\{{.*\}}", raw, re.DOTALL)
        return json.loads(m.group(0))
    except Exception:
        return {
            "summary": raw,
            "risks": [],
            "recommendations": [],
            "actions": [],
        }
