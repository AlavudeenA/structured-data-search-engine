import os
import requests

headers = {
    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"
}

r = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers=headers
)

print(r.json())