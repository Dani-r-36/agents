import requests

url = "http://localhost:8080/generate"
payload = {
    "prompts": [
        "1 + 1 equals:",
        "The capital of Japan is",
    ]
}

print("--> Sending request to Docker container...", flush=True)

try:
    response = requests.post(url, json=payload, timeout=300)
    print(f"--> Response received! Status Code: {response.status_code}", flush=True)

    if response.status_code == 200:
        data = response.json()
        print("\n=== GENERATED RESULTS ===", flush=True)
        for item in data["outputs"]:
            print("-" * 70, flush=True)
            print(f"Prompt: {item['prompt']}\n", flush=True)
            print(f"Output: {item['output']}\n", flush=True)
    else:
        print(f"Request failed with status: {response.status_code}", flush=True)
        print(response.text, flush=True)

except requests.exceptions.Timeout:
    print("Error: The request timed out while waiting for output.", flush=True)
except Exception as e:
    print(f"An error occurred: {e}", flush=True)