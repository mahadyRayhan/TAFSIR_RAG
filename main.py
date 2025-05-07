from flask import Flask, request, jsonify
import time
import os
from hadid_v3 import TafsirRAG # Import the updated class
import glob

# --- Global Initialization ---
print("Starting Flask application...")
start_init_time = time.time()
rag_system = None # Initialize as None
try:
    # Ensure the path pattern correctly points to your PDFs
    doc_pattern = "data/ibn_kathir/*.pdf"
    # if not os.path.exists("data") or not glob.glob(doc_pattern):
    #      print(f"\nWARNING: Document directory 'data' or PDFs matching '{doc_pattern}' not found.")
    #      print("Please ensure the directory exists and contains PDF files relative to main.py")
    #      # Depending on requirements, you might exit or continue without a functional RAG
    #      # raise FileNotFoundError("Required document path not found.")

    # # Create a single instance - runs __init__ once
    # rag_system = TafsirRAG(
    #         vector_db_type="chroma", # Or "csv", "inmemory"
    #         log_level="INFO",        # Or "DEBUG"
    #         documents_path=doc_pattern,
    #         # vector_db_path="my_custom_hadith_db.csv" # Optional: Custom path
    #         use_embedding_cache=True, # Enable caching
    #         use_llm_cache=True
    #     )
    try:
        # Ensure the path pattern correctly points to your PDFs
        doc_pattern = "data/*.pdf"  # <--- THIS IS THE CURRENT PATH IN YOUR LOGS
        if not os.path.exists("data") or not glob.glob(doc_pattern): # Or the folder your PDF is in
            print(f"\nWARNING: Document directory 'data' or PDFs matching '{doc_pattern}' not found.")
            # ...
        rag_system = TafsirRAG(
                vector_db_type="chroma",
                log_level="INFO",
                documents_path=doc_pattern, # <--- THIS VALUE IS BEING USED
                use_embedding_cache=True,
                use_llm_cache=True
            )
    except FileNotFoundError as e:
        print(f"FATAL: Required document path not found: {e}")
        # Handle the error as needed, e.g., exit or set rag_system to None
        rag_system = None
    end_init_time = time.time()
    print(f"TafsirRAG initialization completed in {end_init_time - start_init_time:.2f} seconds.")

# Handle potential errors during initialization more gracefully
except FileNotFoundError as e:
     print(f"FATAL: Failed to initialize TafsirRAG - Required document path not found: {e}")
     # rag_system remains None
except ImportError as e:
     print(f"FATAL: Failed to initialize TafsirRAG - Missing dependency: {e}")
     print("Please ensure all required libraries (like chromadb if used) are installed.")
     # rag_system remains None
except Exception as e:
     print(f"FATAL: An unexpected error occurred during TafsirRAG initialization: {e}")
     import traceback
     traceback.print_exc()
     # rag_system remains None

app = Flask(__name__)


@app.route('/query', methods=['POST'])
def handle_query():
    """Handles incoming queries by calling the RAG system."""
    if rag_system is None:
         # Use logger if available
         logger = getattr(rag_system, 'logger', logging.getLogger())
         logger.error("API request failed: RAG system object is not initialized.")
         return jsonify({"error": "RAG system is not available. Initialization failed."}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    # Optional: Get parameters from request
    top_k = data.get('top_k', 5)
    max_hops = data.get('max_hops', 3)
    request_eval = data.get('evaluate', False)
    generate_graph = data.get('generate_graph', True) # Add graph flag, default True

    rag_system.logger.info(f"Received API query: '{question}' (top_k={top_k}, max_hops={max_hops}, eval={request_eval}, graph={generate_graph})")

    try:
        # Call the process_query method, including the new flag
        result_dict = rag_system.process_query(
            question=question,
            top_k=top_k,
            max_hops=max_hops,
            request_evaluation=request_eval,
            generate_graph=generate_graph # Pass the flag
        )

        # Construct the API response from the result dictionary
        api_response = {
            "query": question,
            "answer": result_dict.get('final_answer', 'Error: No answer generated.'),
            "source_info": result_dict.get('source_info', ''),
            "confidence": result_dict.get('confidence_score'),
            "formatted_reasoning": result_dict.get('formatted_reasoning'), # Add formatted trace
            "evaluation": result_dict.get('evaluation_metrics') if request_eval else None,
            "debug": result_dict.get('debug_info') # Contains graph_filename
        }

        # Remove keys with None values for a cleaner response
        api_response = {k: v for k, v in api_response.items() if v is not None}

        return jsonify(api_response)

    except Exception as e:
        logger = getattr(rag_system, 'logger', logging.getLogger())
        logger.exception(f"API Error during RAG processing for query '{question}': {e}")
        return jsonify({"error": f"An internal error occurred processing your query."}), 500


# --- Run Flask App (for local testing) ---
if __name__ == '__main__':
    # Make sure initialization didn't fail critically before running
    if rag_system is None:
         print("\nCRITICAL ERROR: TafsirRAG system failed to initialize. Flask app cannot start.")
         print("Please check errors above, ensure dependencies are installed, and document paths are correct.")
    else:
         print("Starting Flask development server...")
         # Use 0.0.0.0 to allow connections from other machines (like your test script if not local)
         # Use a port that's likely free, e.g., 5001
         server_port = int(os.environ.get("PORT", 5001))
         print(f"Flask server running on http://0.0.0.0:{server_port}")
         app.run(debug=False, host='0.0.0.0', port=server_port) # Turn debug=False for cleaner logs in production/testing