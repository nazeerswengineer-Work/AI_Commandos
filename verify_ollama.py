import requests
import json

OLLAMA_URL = "http://localhost:11434"

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def check_ollama():
    print_header("OLLAMA CONNECTIVITY")

    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)

        if response.status_code == 200:
            print("SUCCESS: Ollama reachable")

            data = response.json()

            models = [
                model["name"]
                for model in data.get("models", [])
            ]

            print("\nAvailable Models:")

            for model in models:
                print(" -", model)

            return models

        else:
            print("FAILED: Unexpected status code")
            print(response.status_code)
            return []

    except Exception as e:
        print("FAILED:", e)
        return []

def test_llm_generation():
    print_header("LLM GENERATION TEST")

    payload = {
        "model": "llama3.2:1b",
        "prompt": "Reply with exactly: REQTRACE_AI_OK",
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60
        )

        data = response.json()

        print(data.get("response", ""))

    except Exception as e:
        print("FAILED:", e)

def test_embedding():
    print_header("EMBEDDING TEST")

    payload = {
        "model": "nomic-embed-text",
        "prompt": "REQTRACE AI embedding verification"
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json=payload,
            timeout=60
        )

        data = response.json()

        embedding = data.get("embedding", [])

        print("Embedding Length:", len(embedding))

    except Exception as e:
        print("FAILED:", e)

if __name__ == "__main__":
    check_ollama()
    test_llm_generation()
    test_embedding()