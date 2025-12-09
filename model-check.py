# Check the version of Google model
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY =  os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set")

url  = "https://generativelanguage.googleapis.com/v1/models"
params = {"key": API_KEY}

resp= requests.get(url, params=params,timeout=20)
resp.raise_for_status()
data = resp.json()
models = data.get("models", [])
print(f" found {len(models)} Aailable Models:")
for m in models:
    name = m.get("name")
    display = m.get("displayName", "N/A")
    input_limit = m.get("inputTokenLimit", "N/A")
    output_limit = m.get("outputTokenLimit", "N/A") 
    print(f"- {m} (Display: {display}, Input limit: {input_limit}, Output limit: {output_limit})")

