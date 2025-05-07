import requests
import json
import os
import time
import glob # Import glob

# --- Configuration ---
# Use the port specified in main.py
DEFAULT_PORT = 5001
API_PORT = os.environ.get("PORT", DEFAULT_PORT)
API_BASE_URL = os.environ.get("API_URL", f"http://127.0.0.1:{API_PORT}")
QUERY_ENDPOINT = f"{API_BASE_URL}/query"

def test_query(question: str, params: dict = None):
    """Sends a question to the API and prints the response."""
    print("-" * 50)
    print(f"Testing question: '{question}'")
    payload = {"question": question}
    if params:
        print(f"With parameters: {params}")
        payload.update(params)

    headers = {"Content-Type": "application/json"}

    try:
        start_time = time.time()
        response = requests.post(QUERY_ENDPOINT, headers=headers, json=payload, timeout=300)
        end_time = time.time()

        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")

        if response.ok: # Check if status code is 2xx
             try:
                 response_data = response.json()
                 print("\nAPI Response JSON:")
                 # Print selected fields first for clarity
                 print(f"  Query: {response_data.get('query')}")
                 print(f"  Answer: {response_data.get('answer')}")
                 print(f"  Confidence: {response_data.get('confidence')}")
                 print(f"  Source Info: {response_data.get('source_info')}")
                 graph_file = response_data.get('debug', {}).get('graph_filename')
                 if graph_file:
                      print(f"  Graph File: {graph_file}")
                 else:
                      print(f"  Graph File: Not generated or generation failed.")

                 # Print the formatted reasoning trace
                 print("\nFormatted Reasoning Trace:")
                 print(response_data.get('formatted_reasoning', 'No reasoning trace provided.'))

                 # Optionally print the full debug info
                 # print("\nDebug Info:")
                 # print(json.dumps(response_data.get('debug'), indent=2))

             except json.JSONDecodeError:
                 print("\nResponse content (not JSON):")
                 print(response.text)
             except Exception as e:
                 print(f"\nError processing response JSON: {e}")
        else:
             print("\nError Response:")
             try:
                 print(json.dumps(response.json(), indent=2))
             except json.JSONDecodeError:
                 print("response:",  response.text)


    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {e}")
        print("Is the Flask server (main.py) running and accessible?")

    print("-" * 50 + "\n") # Add a newline for better readability


# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Starting RAG API Tests ---")
    print(f"Targeting API Endpoint: {QUERY_ENDPOINT}")

    test_questions = [
        "Explain the teaching of Surah Al-Fatiha according to Ibn Kathir.",
        "Based on Ibn Kathir's Tafsir of the early sections of Surah Al-Baqarah, contrast the primary characteristics he attributes to the believers (al-muttaqin) with those he describes for the hypocrites (al-munafiqin). What are the key consequences or warnings Ibn Kathir highlights regarding hypocrisy?",
        "From Ibn Kathir's explanation of the story of Adam and Iblis (Satan) in Surah Al-Baqarah, what were the core reasons for Iblis's disobedience, and what are the primary lessons or admonitions Ibn Kathir derives from this specific narrative for mankind?"
    ]

    # --- Run Tests ---
    for q in test_questions:
        # Example: Request graph generation (default is True anyway)
        test_query(q, params={"generate_graph": True})
        # Example: Disable graph generation
        # test_query(q, params={"generate_graph": False})
        time.sleep(1)

    print("--- Tests Finished ---")