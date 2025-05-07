# import requests
# import json
# import os
# import time

# # --- Configuration ---
# # Use the port specified in main.py's if __name__ == '__main__': block
# API_BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:5001")
# QUERY_ENDPOINT = f"{API_BASE_URL}/query"

# # --- Test Function ---
# def test_query(question: str):
#     """Sends a question to the API and prints the response."""
#     print("-" * 40)
#     print(f"Testing question: '{question}'")

#     payload = {"question": question}
#     headers = {"Content-Type": "application/json"}

#     try:
#         start_time = time.time()
#         response = requests.post(QUERY_ENDPOINT, headers=headers, json=payload, timeout=180) # Increased timeout for potentially long RAG calls
#         end_time = time.time()

#         print(f"Status Code: {response.status_code}")
#         print(f"Response Time: {end_time - start_time:.2f} seconds")

#         try:
#             response_data = response.json()
#             print("Response JSON:")
#             # Pretty print the JSON response
#             print(json.dumps(response_data['answer'], indent=2))
#         except json.JSONDecodeError:
#             print("Response content (not JSON):")
#             print(response.text)

#     except requests.exceptions.RequestException as e:
#         print(f"Request failed: {e}")

#     print("-" * 40)
#     print("\n") # Add a newline for better readability between tests


# # --- Main Execution ---
# if __name__ == "__main__":
#     # Ensure the Flask app (main.py) is running before executing this script.
#     print(f"Starting tests against API: {QUERY_ENDPOINT}")

#     test_questions = [
#         # "Explain Hadith No. 1",
#         "what did Ibn Rajab, writing on the subject of prejudice?",
#         "What are the functions of niyyah?",
#         # "What is ihsan?",
#     ]

#     for q in test_questions:
#         test_query(q)
#         time.sleep(1) # Small delay between requests

#     print("Tests finished.")

# test.py (Updated)

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
                 print(response.text)


    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {e}")
        print("Is the Flask server (main.py) running and accessible?")

    print("-" * 50 + "\n") # Add a newline for better readability


# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Starting RAG API Tests ---")
    print(f"Targeting API Endpoint: {QUERY_ENDPOINT}")
    # ... (Connectivity check remains the same) ...

    test_questions = [
        "what did Ibn Rajab, writing on the subject of prejudice?",
        "What are the functions of niyyah?"
    ]

    # --- Run Tests ---
    for q in test_questions:
        # Example: Request graph generation (default is True anyway)
        test_query(q, params={"generate_graph": True})
        # Example: Disable graph generation
        # test_query(q, params={"generate_graph": False})
        time.sleep(1)

    print("--- Tests Finished ---")