from flask import Flask, request, jsonify, render_template, send_from_directory # Added render_template, send_from_directory
import time
import os
from tafsir import TafsirRAG # Import the updated class
import glob
import logging # Ensure logging is imported

# --- Global Initialization ---
print("Starting Flask application...")
start_init_time = time.time()
rag_system = None # Initialize as None
# Using the exact doc_pattern and initialization logic from your original file
doc_pattern_original = "data/source/ibn_kathir/*.pdf"

try:
    # Ensure the path pattern correctly points to your PDFs
    # The nested try-except for initialization is specific to your original file, preserving it.
    try:
        # Check if the directory exists before globbing, as in your original
        pdf_directory_original = os.path.dirname(doc_pattern_original) # "data/source/ibn_kathir"
        if not os.path.exists(pdf_directory_original) :
            print(f"\nWARNING: Document directory '{pdf_directory_original}' for PDFs matching '{doc_pattern_original}' not found.")
            # rag_system will remain None as per original logic flow if path doesn't exist.
        elif not glob.glob(doc_pattern_original):
             print(f"\nWARNING: No PDFs found matching '{doc_pattern_original}' in directory '{pdf_directory_original}'.")
             # rag_system will remain None
        
        # Proceed with initialization attempt regardless, to match original behavior
        # where TafsirRAG itself might raise FileNotFoundError if paths are ultimately bad.
        rag_system = TafsirRAG(
                vector_db_type="csv", #"chroma",
                log_level="INFO",
                documents_path=doc_pattern_original, # Using your original pattern
                vector_db_path="data/legacy/vector_db_hadith.csv",
                feedback_db_path="data/legacy/feedback_history.csv",
                log_file="logs/hadith_rag.log",
                use_embedding_cache=True,
                use_llm_cache=True
            )
    except FileNotFoundError as e: # This is the inner FileNotFoundError
        print(f"FATAL (inner try): Required document path not found during TafsirRAG init: {e}")
        rag_system = None # Ensure rag_system is None
    
    # This part is reached if the inner try for TafsirRAG instantiation didn't raise an unhandled exception
    # or if it successfully initialized.
    if rag_system:
        end_init_time = time.time()
        print(f"TafsirRAG initialization completed in {end_init_time - start_init_time:.2f} seconds.")
    elif not os.path.exists(os.path.dirname(doc_pattern_original)): # If rag_system is still None due to path warning
        print(f"TafsirRAG initialization SKIPPED because directory '{os.path.dirname(doc_pattern_original)}' was not found.")
    elif not glob.glob(doc_pattern_original):
        print(f"TafsirRAG initialization SKIPPED because no PDFs found matching '{doc_pattern_original}'.")


# Outer exception handling from your original file
except FileNotFoundError as e: # This catches FileNotFoundError if raised by TafsirRAG directly to the outer block
     print(f"FATAL (outer try): Failed to initialize TafsirRAG - Required document path not found: {e}")
     rag_system = None
except ImportError as e:
     print(f"FATAL: Failed to initialize TafsirRAG - Missing dependency: {e}")
     print("Please ensure all required libraries (like chromadb if used) are installed.")
     rag_system = None
except Exception as e:
     print(f"FATAL: An unexpected error occurred during TafsirRAG initialization: {e}")
     import traceback
     traceback.print_exc()
     rag_system = None

app = Flask(__name__)

# --- Routes for UI and Static Files ---
@app.route('/')
def index():
    """Serves the main HTML page for the UI."""
    return render_template('index.html') # Assumes index.html is in 'templates' folder

@app.route('/rag_graphs/<path:filename>')
def serve_graph_image(filename):
    """Serves graph images from the 'rag_graphs' directory."""
    # The 'rag_graphs' directory is expected to be at the same level as main.py
    # or where the tafsir.py script creates it.
    graph_dir = "rag_graphs" 
    if not os.path.exists(graph_dir):
        # Optionally log that the directory doesn't exist yet
        logger = getattr(rag_system, 'logger', logging.getLogger('flask.app'))
        logger.warning(f"Attempted to serve graph from '{graph_dir}', but directory does not exist.")
        return "Graph directory not found.", 404
    return send_from_directory(graph_dir, filename)

@app.route('/history', methods=['GET'])
def get_history():
    current_logger = logging.getLogger('app.history')
    if rag_system and hasattr(rag_system, 'logger'):
        current_logger = rag_system.logger
    # ... (add more robust logger initialization if needed, similar to handle_query)

    if rag_system is None or rag_system.feedback_history is None:
        current_logger.error("API request for /history failed: RAG system or feedback_history not initialized.")
        return jsonify({"error": "History is not available."}), 503

    try:
        # Extract only 'query' and 'answer' from each feedback record
        simplified_history = []
        for record in rag_system.feedback_history:
            # print(record.get("query"), record.get("answer"), record.get("reasoning_trace").split('->')[-1])  # Debug print
            CONFIDENCE = record.get("reasoning_trace").split('->')[-1] if record.get("reasoning_trace") else 0
            if 'Confidence Score' not in CONFIDENCE:
                CONFIDENCE = 0
            simplified_history.append({
                "query": record.get("query"),  # UPDATED: Key changed from "question" to "query"
                "answer": record.get("answer"),
                "confidence": CONFIDENCE
            })
        
        # The history is typically shown in reverse chronological order (newest first)
        return jsonify(simplified_history[::-1]) 

    except Exception as e:
        current_logger.exception(f"API Error during /history processing: {e}")
        return jsonify({"error": f"An internal error occurred retrieving history: {str(e)}"}), 500

@app.route('/get_resource_files', methods=['GET'])
def get_resource_files():
    """Endpoint to list the PDF resource files."""
    current_logger = getattr(rag_system, 'logger', logging.getLogger('flask.app'))
    try:
        # doc_pattern_original is defined globally: "data/source/ibn_kathir/*.pdf"
        pdf_files = glob.glob(doc_pattern_original)
        resource_filenames = sorted([os.path.basename(f) for f in pdf_files]) # Sort for consistent order
        
        if not resource_filenames:
            current_logger.info(f"No resource files found matching pattern: {doc_pattern_original}")
        return jsonify({"files": resource_filenames})
    except Exception as e:
        current_logger.error(f"Error listing resource files for pattern '{doc_pattern_original}': {e}")
        return jsonify({"error": "Could not retrieve resource files."}), 500
    
# --- Your Original /query Route ---
@app.route('/query', methods=['POST'])
def handle_query():
    """Handles incoming queries by calling the RAG system."""
    # Using the logger fetching logic from your original file
    # Default to a basic logger if rag_system or its logger is not available
    current_logger = logging.getLogger('app.query') # Fallback logger
    if rag_system and hasattr(rag_system, 'logger'):
        current_logger = rag_system.logger
    elif rag_system is None: # If RAG system itself is None, use a basic configured logger
        if not current_logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            current_logger.addHandler(handler)
            current_logger.setLevel(logging.INFO)


    if rag_system is None:
         current_logger.error("API request failed: RAG system object is not initialized.")
         return jsonify({"error": "RAG system is not available. Initialization failed."}), 503

    if not request.is_json:
        current_logger.warning("Invalid request to /query: Request must be JSON.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query_text = data.get('query') # UPDATED: Expected key is now 'query'
    if not query_text:
        current_logger.warning("Invalid request to /query: Missing 'query' in request body.") # UPDATED: Message
        return jsonify({"error": "Missing 'query' in request body"}), 400 # UPDATED: Message

    # Optional: Get parameters from request
    top_k = data.get('top_k', 5)
    max_hops = data.get('max_hops', 3)
    request_eval = data.get('evaluate', False)
    generate_graph = data.get('generate_graph', False)

    current_logger.info(f"Received API query: '{query_text}' (top_k={top_k}, max_hops={max_hops}, eval={request_eval}, graph={generate_graph})")

    try:
        # Call the process_query method
        result_dict = rag_system.process_query(
            query=query_text, # UPDATED: Parameter name is 'query'
            top_k=top_k,
            max_hops=max_hops,
            request_evaluation=request_eval,
            generate_graph=generate_graph
        )

        # Construct the API response
        api_response = {
            "query": query_text, # UPDATED: Value comes from query_text
            "answer": result_dict.get('final_answer', 'Error: No answer generated.'),
            "source_info": result_dict.get('source_info', ''),
            "confidence": result_dict.get('confidence_score'),
            "formatted_reasoning": result_dict.get('formatted_reasoning'),
            "evaluation": result_dict.get('evaluation_metrics') if request_eval else None,
            "debug": result_dict.get('debug_info')
        }
        api_response = {k: v for k, v in api_response.items() if v is not None}
        return jsonify(api_response)

    except Exception as e:
        current_logger.exception(f"API Error during RAG processing for query '{query_text}': {e}")
        return jsonify({"error": f"An internal error occurred processing your query: {str(e)}"}), 500


# --- Run Flask App (for local testing) ---
if __name__ == '__main__':
    # Using the exact startup conditions from your original file
    if rag_system is None:
         print("\nCRITICAL ERROR: TafsirRAG system failed to initialize. Flask app cannot start.")
         print("Please check errors above, ensure dependencies are installed, and document paths are correct.")
         print(f"  Specifically, check the PDF path: '{doc_pattern_original}' and its parent directory.")
    else:
         print("Starting Flask development server...")
         server_port = int(os.environ.get("PORT", 5001))
         
         # Check for templates directory
         if not os.path.exists("templates"):
             print("WARNING: 'templates' directory not found. Create it and place 'index.html' inside for the UI to work.")
         # Check for rag_graphs directory (it's created by tafsir.py, so just inform)
         if not os.path.exists("rag_graphs"):
             print("INFO: 'rag_graphs' directory not found. It should be created by the RAG system when 'generate_graph' is true.")

         print(f"Flask server running. Open your browser to http://127.0.0.1:{server_port}/")
         # threaded=True can be helpful for development to handle multiple simultaneous requests better from the UI
         app.run(debug=False, host='0.0.0.0', port=server_port, threaded=True)