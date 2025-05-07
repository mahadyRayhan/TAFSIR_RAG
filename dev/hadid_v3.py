import os
import glob
import re
import time
import logging
import numpy as np
import pandas as pd
import uuid
from graphviz import Digraph
from typing import Any, Dict, List, Tuple, Optional, Union
import PyPDF2
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from difflib import SequenceMatcher # For simple deduplication

# Load environment variables
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=API_KEY)

# Default Model IDs (can be overridden in __init__)
DEFAULT_MODEL_ID = "gemini-2.0-flash" # Changed to a more recent model, adjust if needed
DEFAULT_TEXT_EMBEDDING_MODEL = "text-embedding-004"

# Default Paths (can be overridden in __init__)
DEFAULT_VECTOR_DB_PATH = "vector_db_hadith.csv"
DEFAULT_DOCUMENTS_PATH_PATTERN = "ibn_kathir/*.pdf" # Ensure this path is correct
DEFAULT_LOG_FILE_PATH = "hadith_rag.log"
DEFAULT_FEEDBACK_DB_PATH = "feedback_history.csv"

class HadidRAG:
    """
    Enhanced RAG system for Imam Nawawi's 40 Hadith.
    Handles resource loading, query processing, multi-hop reasoning,
    evaluation, feedback, and optimized vector storage options.
    """
    def __init__(self,
                 model_id: str = DEFAULT_MODEL_ID,
                 embedding_model_id: str = DEFAULT_TEXT_EMBEDDING_MODEL,
                 vector_db_type: str = "csv", # Options: "csv", "chroma", "inmemory"
                 vector_db_path: str = DEFAULT_VECTOR_DB_PATH,
                 documents_path: str = DEFAULT_DOCUMENTS_PATH_PATTERN,
                 log_level: str = "INFO",
                 log_file: str = DEFAULT_LOG_FILE_PATH,
                 feedback_db_path: str = DEFAULT_FEEDBACK_DB_PATH,
                 chunk_strategy: str = "recursive", # Future: "semantic"
                 chunk_size: int = 1500,
                 chunk_overlap: int = 200,
                 use_embedding_cache: bool = True,
                 use_llm_cache: bool = True):
        """
        Initialize the HadidRAG system.

        Args:
            model_id: Generative model ID for text generation.
            embedding_model_id: Text embedding model ID.
            vector_db_type: Type of vector database ("csv", "chroma", "inmemory").
            vector_db_path: Path for CSV vector database file.
            documents_path: Glob pattern for source PDF documents.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
            log_file: Path for the log file.
            feedback_db_path: Path for the feedback CSV file.
            chunk_strategy: Strategy for text chunking ("recursive").
            chunk_size: Target size for text chunks.
            chunk_overlap: Overlap between consecutive chunks.
            use_embedding_cache: Enable in-memory caching for embeddings.
            use_llm_cache: Enable in-memory caching for LLM responses.
        """
        self._setup_logging(log_level, log_file)
        self.logger.info(f"--- Initializing HadidRAG (PID: {os.getpid()}) ---")
        self.logger.info(f"Using Generative Model: {model_id}")
        self.logger.info(f"Using Embedding Model: {embedding_model_id}")
        self.logger.info(f"Vector DB Type: {vector_db_type}")
        self.logger.info(f"Chunk Strategy: {chunk_strategy} (Size: {chunk_size}, Overlap: {chunk_overlap})")

        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        self.vector_db_type = vector_db_type.lower()
        self.vector_db_path = vector_db_path
        self.documents_path_pattern = documents_path
        self.feedback_db_path = feedback_db_path
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Caching
        self.use_embedding_cache = use_embedding_cache
        self.embedding_cache: Dict[str, List[float]] = {}
        self.use_llm_cache = use_llm_cache
        self.llm_cache: Dict[str, str] = {}

        # Feedback
        self.feedback_history: List[Dict[str, Any]] = []

        try:
            self._initialize_clients()
            self._initialize_vector_db()
            self._load_feedback_history()
            self.logger.info("--- HadidRAG Initialized Successfully ---")
        except Exception as e:
            self.logger.exception(f"CRITICAL ERROR during HadidRAG Initialization: {e}")
            raise # Re-raise after logging

    def _setup_logging(self, log_level_str: str, log_file: str):
        """Set up proper logging instead of print statements."""
        numeric_level = getattr(logging, log_level_str.upper(), None)
        if not isinstance(numeric_level, int):
            logging.basicConfig(level=logging.INFO) # Default if level is invalid
            logging.warning(f'Invalid log level: {log_level_str}. Defaulting to INFO.')
            numeric_level = logging.INFO

        self.logger = logging.getLogger("HadithRAG")
        self.logger.setLevel(numeric_level)
        self.logger.propagate = False # Prevent duplicate logs in root logger

        # Clear existing handlers if re-initializing
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler for persistent logs
        try:
            fh = logging.FileHandler(log_file, mode='a') # Append mode
            fh.setLevel(numeric_level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        except Exception as e:
             self.logger.error(f"Failed to set up file handler at {log_file}: {e}")

        self.logger.info(f"Logging initialized at level {log_level_str.upper()} to console and {log_file}")

    def _format_reasoning_trace(self, reasoning_trace: List[str]) -> str:
        """Formats the reasoning trace list into a more readable string."""
        formatted = ["**Reasoning Process:**\n" + "-"*20]
        hop_details = []
        current_hop = 0

        for step in reasoning_trace:
            step = step.strip()
            if step == "START":
                formatted.append("1. Initial Query Transformation") # Assuming transform runs before first hop
                continue
            elif step.startswith("--- Hop"):
                if hop_details: # Format previous hop before starting new one
                     formatted.append(f"\n**Hop {current_hop}:**")
                     formatted.extend([f"  - {d}" for d in hop_details])
                     hop_details = [] # Reset for next hop
                current_hop = int(step.split(" ")[2]) if len(step.split(" ")) > 2 else current_hop + 1
            elif step.startswith("Retrieving with query"):
                 hop_details.append(f"Query: {step.split('->')[-1].strip()}")
            elif step.startswith("Retrieved"):
                 hop_details.append(f"Retrieval: {step.split(':')[-1].strip()}")
            elif step.startswith("Added"):
                 hop_details.append(f"Context Mgmt: {step.split(':')[-1].strip()}")
            elif step.startswith("Reasoning result"):
                 action_value = step.split('->')[-1].strip()
                 hop_details.append(f"Reasoning Action: {action_value}")
            elif step.startswith("Reasoning ->"): # Summarizes the decision
                 pass # Often redundant with "Reasoning Action", skip for cleaner output
            elif step == "--- Final Answer Generation ---":
                 if hop_details: # Format last hop
                      formatted.append(f"\n**Hop {current_hop}:**")
                      formatted.extend([f"  - {d}" for d in hop_details])
                 formatted.append("\n**Final Answer Generation**")
            elif step.startswith("Confidence Score:") or step.startswith("Evaluation Metrics:"):
                 formatted.append(f"  - {step}")
            elif step.startswith("Max hops"):
                 hop_details.append(step) # Add this as a note in the last hop
            else:
                 # Add other steps if they don't fit known patterns
                 if current_hop > 0 and step not in ["START", "--- Final Answer Generation ---"]:
                      hop_details.append(f"Step: {step}")


        # Append details of the last hop if any remain
        if hop_details and not formatted[-1].startswith("**Hop"):
             formatted.append(f"\n**Hop {current_hop}:**")
             formatted.extend([f"  - {d}" for d in hop_details])


        formatted.append("-"*20)
        return "\n".join(formatted)
    
    def _generate_hop_graph(self, reasoning_trace: List[str], query_history: List[str], graph_dir: str = "rag_graphs") -> Optional[str]:
        """Generates a Graphviz graph visualizing the RAG hops."""
        if not reasoning_trace:
            return None

        try:
            os.makedirs(graph_dir, exist_ok=True)
            graph_id = str(uuid.uuid4())[:8]
            filename = os.path.join(graph_dir, f"rag_hop_graph_{graph_id}")

            dot = Digraph(comment='RAG Multi-Hop Process', format='png')
            dot.attr(rankdir='TB', size='10,10', label=f'RAG Process Flow (ID: {graph_id})', fontsize='20')
            dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontsize='10')
            dot.attr('edge', fontsize='8')

            node_counter = 0
            current_node_name = f"n{node_counter}" # Stores the name of the current node
            final_node_created = False  # Flag to track if the final node was added

            start_query = query_history[0] if query_history else "Initial Query (N/A)"
            # Ensure label is not excessively long
            start_label = f"Start\nQuery:\n{start_query[:150]}{'...' if len(start_query)>150 else ''}"
            dot.node(current_node_name, start_label)
            node_counter += 1

            hop_num = 0
            current_query_index = 0 # Index into query_history

            for i, step in enumerate(reasoning_trace):
                if final_node_created: # Stop adding nodes after final answer node is created
                     break
                step = step.strip()

                if step.startswith("--- Hop"):
                    hop_num = int(step.split(" ")[2]) if len(step.split(" ")) > 2 else hop_num + 1
                    current_query_index += 1
                    query_text = query_history[current_query_index] if current_query_index < len(query_history) else f"Query for Hop {hop_num} (N/A)"
                    query_label = f"Hop {hop_num}\nQuery:\n{query_text[:150]}{'...' if len(query_text)>150 else ''}"

                    query_node_name = f"n{node_counter}"
                    dot.node(query_node_name, query_label, fillcolor='lightgreen')
                    dot.edge(current_node_name, query_node_name, label=f"Start Hop {hop_num}")
                    current_node_name = query_node_name
                    node_counter += 1

                elif step.startswith("Reasoning result"):
                    reasoning_node_name = f"n{node_counter}"
                    action = step.split("Action='")[-1].split("'")[0]
                    value_part = step.split("Value='")[-1]
                    value = value_part[:70] + ("..." if len(value_part) > 70 else "") # Adjusted length slightly
                    value = value[:-3] if value.endswith("...'") else value
                    reasoning_label = f"Hop {hop_num} Reasoning\nAction: {action}\nValue: '{value}'"

                    dot.node(reasoning_node_name, reasoning_label, fillcolor='lightyellow')
                    dot.edge(current_node_name, reasoning_node_name, label="Process Context")
                    current_node_name = reasoning_node_name
                    node_counter += 1

                    # If ANSWER_COMPLETE or ERROR, link to final answer node and set flag
                    if action == "ANSWER_COMPLETE" or action == "ERROR":
                        final_node_name = f"n{node_counter}"
                        dot.node(final_node_name, "Final Answer Generation", shape='ellipse', fillcolor='orange')
                        dot.edge(current_node_name, final_node_name, label=action)
                        current_node_name = final_node_name # Update current node
                        final_node_created = True  # Set the flag
                        node_counter += 1
                        # break # Optional: Stop processing trace immediately after adding final node

            # --- Revised Final Connection Logic ---
            # After the loop, if the final answer node wasn't created during the loop (e.g. max hops reached)
            # add it now and connect the last processed node to it.
            if not final_node_created:
                final_node_name = f"n{node_counter}"
                dot.node(final_node_name, "Final Answer Generation", shape='ellipse', fillcolor='orange')
                # Determine the label for the final edge
                label = "Max Hops Reached" if any("Max hops reached" in s for s in reasoning_trace) else "Proceed to Final"
                # Ensure current_node_name is valid before adding the edge
                if current_node_name and current_node_name.startswith("n"):
                     dot.edge(current_node_name, final_node_name, label=label)
                else:
                     # This case should be rare, but handles if the trace was very short/empty
                     self.logger.warning("Graph generation: Could not determine valid node to connect to final answer.")


            # Render the graph
            output_path = dot.render(filename=filename, view=False, cleanup=True)
            self.logger.info(f"Generated RAG hop graph: {output_path}")
            return output_path

        except ImportError:
            self.logger.warning("Graphviz Python library not found. Cannot generate graph. Please run 'pip install graphviz'.")
            return None
        except FileNotFoundError:
             self.logger.error("Graphviz executable not found in PATH. Cannot generate graph. Please install Graphviz (see https://graphviz.org/download/).")
             return None
        except Exception as e:
            # Log the full traceback for unexpected errors during graph generation
            self.logger.exception(f"Failed to generate RAG hop graph: {e}")
            return None
        
    def _initialize_clients(self):
        """Initializes Google AI clients."""
        try:
            # Embedding model name format for the API
            self.embedding_model_name = f"models/{self.embedding_model_id}"

            # Generation model configuration
            generation_config = {
                "temperature": 0.7, # Slightly lower temp for more focused answers
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            self.model = genai.GenerativeModel(
                model_name=self.model_id,
                generation_config=generation_config,
            )
            self.logger.info("Google AI Clients Initialized.")
        except Exception as e:
            self.logger.error(f"Error initializing Google AI clients: {e}")
            raise

    def _initialize_vector_db(self):
        """Initialize the appropriate vector database based on configuration."""
        self.logger.info(f"Initializing vector database (Type: {self.vector_db_type})...")
        if self.vector_db_type == "chroma":
            self._setup_chroma_db()
        elif self.vector_db_type == "csv":
            self._load_or_build_csv_db()
        elif self.vector_db_type == "inmemory":
            self._load_or_build_inmemory_db()
        else:
            self.logger.error(f"Unknown vector database type: {self.vector_db_type}")
            raise ValueError(f"Unknown vector database type: {self.vector_db_type}")
        self.logger.info("Vector database initialization complete.")


    def _setup_chroma_db(self):
        """Set up ChromaDB for vector storage."""
        try:
            import chromadb
            from chromadb.config import Settings

            # Initialize ChromaDB client (persistent or in-memory)
            # For persistence, specify a path: path="path/to/chroma_data"
            self.chroma_client = chromadb.Client(Settings(
                 # chroma_db_impl="duckdb+parquet", # Optional: specify implementation
                 # persist_directory=".chromadb_cache" # Optional: Enable persistence
            ))


            # Create or get collection
            collection_name = "hadith_collection"
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                # metadata={"hnsw:space": "cosine"} # Use cosine distance
            )
            self.logger.info(f"ChromaDB client connected. Using collection: '{collection_name}'")

            # Check if we need to populate
            if self.collection.count() == 0:
                self.logger.info("ChromaDB collection is empty. Attempting to build/migrate...")
                # Option 1: Migrate from existing CSV if it exists
                if os.path.exists(self.vector_db_path):
                    self.logger.info(f"Found existing CSV at '{self.vector_db_path}'. Migrating to ChromaDB...")
                    try:
                        csv_db = pd.read_csv(self.vector_db_path)
                        self._populate_chroma_from_dataframe(csv_db)
                    except Exception as migrate_err:
                        self.logger.error(f"Error migrating from CSV to ChromaDB: {migrate_err}. Trying to build from documents.")
                        self._build_index_and_populate_vector_db() # Fallback to building
                # Option 2: Build directly from documents
                else:
                    self.logger.info(f"No existing CSV found at '{self.vector_db_path}'. Building ChromaDB index from documents...")
                    self._build_index_and_populate_vector_db()

            self.logger.info(f"ChromaDB ready. Collection '{collection_name}' contains {self.collection.count()} documents.")

        except ImportError:
            self.logger.error("ChromaDB library not installed. Please run 'pip install chromadb'")
            self.logger.warning("Falling back to CSV storage.")
            self.vector_db_type = "csv" # Change type
            self._load_or_build_csv_db() # Load/Build CSV instead
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during ChromaDB setup: {e}")
            raise

    def _populate_chroma_from_dataframe(self, df: pd.DataFrame):
        """Adds data from a Pandas DataFrame to the ChromaDB collection."""
        if 'embeddings' not in df.columns or 'chunk_text' not in df.columns:
             raise ValueError("DataFrame must contain 'embeddings' and 'chunk_text' columns for ChromaDB population.")

        # Convert string embeddings if needed
        if isinstance(df['embeddings'].iloc[0], str):
            self.logger.info("Converting string embeddings from DataFrame...")
            df['embeddings'] = df['embeddings'].apply(lambda x: list(map(float, re.findall(r"-?\d+\.?\d*", x)))) # More robust parsing
        elif isinstance(df['embeddings'].iloc[0], list):
             pass # Already in list format
        elif isinstance(df['embeddings'].iloc[0], np.ndarray):
             df['embeddings'] = df['embeddings'].apply(lambda x: x.tolist())
        else:
             raise TypeError(f"Unsupported embedding type in DataFrame: {type(df['embeddings'].iloc[0])}")

        # Prepare data for ChromaDB batch insertion
        ids = [f"chunk_{i}" for i in range(len(df))]
        embeddings = df['embeddings'].tolist()
        documents = df['chunk_text'].tolist()
        metadatas = [{
            "document_name": row.get('document_name', 'Unknown'),
            "page_number": int(row.get('page_number', -1)),
            "chunk_number": int(row.get('chunk_number', -1))
        } for _, row in df.iterrows()]

        # Add documents to ChromaDB in batches
        batch_size = 100 # ChromaDB recommends batches of reasonable size
        total_added = 0
        for i in range(0, len(df), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_documents = documents[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]

            if not batch_ids: continue # Skip empty batches

            try:
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_ids)
                self.logger.info(f"Added batch {i//batch_size + 1} ({len(batch_ids)} documents) to ChromaDB. Total added: {total_added}")
            except Exception as batch_err:
                 self.logger.error(f"Error adding batch {i//batch_size + 1} to ChromaDB: {batch_err}. Skipping batch.")
                 # Optional: Add more robust error handling here (e.g., retry individual items)

        self.logger.info(f"Finished populating ChromaDB from DataFrame. Added {total_added} documents.")


    def _load_or_build_csv_db(self):
        """Loads the vector database from CSV or builds it if not found."""
        self.vector_db_df: Optional[pd.DataFrame] = None # Store DataFrame here
        try:
            if os.path.exists(self.vector_db_path):
                self.logger.info(f"Loading existing vector database from {self.vector_db_path}...")
                self.vector_db_df = pd.read_csv(self.vector_db_path)
                if 'embeddings' in self.vector_db_df.columns:
                    # Check type of first element to decide conversion
                    first_emb = self.vector_db_df['embeddings'].iloc[0]
                    if isinstance(first_emb, str):
                        self.logger.info("Converting string embeddings in CSV to numpy arrays...")
                        # Use a robust parsing method
                        self.vector_db_df['embeddings'] = self.vector_db_df['embeddings'].apply(
                             lambda x: np.array(list(map(float, re.findall(r"-?\d+\.?\d*", x)))) if isinstance(x, str) else np.array([])
                         )
                    elif isinstance(first_emb, list):
                        self.vector_db_df['embeddings'] = self.vector_db_df['embeddings'].apply(np.array)
                    elif isinstance(first_emb, np.ndarray):
                        pass # Already numpy arrays
                    else:
                         self.logger.warning(f"Embeddings column has unexpected type: {type(first_emb)}. Skipping conversion.")

                    # Basic validation
                    if len(self.vector_db_df) > 0 and len(self.vector_db_df['embeddings'].iloc[0]) > 0:
                         self.logger.info(f"CSV Vector database loaded successfully. Shape: {self.vector_db_df.shape}, Embedding Dim: {len(self.vector_db_df['embeddings'].iloc[0])}")
                    else:
                         self.logger.warning(f"CSV Vector database loaded but seems empty or has invalid embeddings. Shape: {self.vector_db_df.shape}")

                else:
                    self.logger.error(f"CSV file '{self.vector_db_path}' is missing the 'embeddings' column.")
                    self.vector_db_df = None # Mark as failed load
            else:
                self.logger.info(f"CSV Vector Database '{self.vector_db_path}' not found. Building index...")
                self.vector_db_df = self._build_index_and_populate_vector_db() # Build and store in df
                if self.vector_db_df is not None and not self.vector_db_df.empty:
                     self._save_dataframe_to_csv(self.vector_db_df, self.vector_db_path)
                else:
                     self.logger.error("Failed to build index, CSV database remains unavailable.")


        except FileNotFoundError as fnf:
            self.logger.error(f"Error during CSV DB loading: {fnf}")
            self.vector_db_df = None
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during CSV DB loading/building: {e}")
            self.vector_db_df = None # Mark as failed

        if self.vector_db_df is None or self.vector_db_df.empty:
             self.logger.critical("CRITICAL ERROR: CSV Vector database could not be loaded or built.")
             # Depending on strictness, you might want to raise an error here
             # raise ValueError("Vector database could not be loaded or built.")

    def _load_or_build_inmemory_db(self):
        """Loads vector database into memory (from CSV if exists, else builds)."""
        self.logger.info("Initializing in-memory vector database...")
        # Attempt to load from CSV first for speed if available
        if os.path.exists(self.vector_db_path):
            self.logger.info(f"Found existing CSV '{self.vector_db_path}', loading into memory.")
            self._load_or_build_csv_db() # This loads into self.vector_db_df
            if self.vector_db_df is not None and not self.vector_db_df.empty:
                self.logger.info("In-memory database populated from CSV.")
            else:
                self.logger.warning("Failed to load from CSV. Attempting to build index directly into memory...")
                self.vector_db_df = self._build_index_and_populate_vector_db()
                if self.vector_db_df is not None and not self.vector_db_df.empty:
                     self.logger.info("In-memory database built successfully from documents.")
                else:
                     self.logger.error("Failed to build index for in-memory database.")
        else:
            # Build directly into memory
            self.logger.info(f"No CSV found at '{self.vector_db_path}'. Building index directly into memory...")
            self.vector_db_df = self._build_index_and_populate_vector_db()
            if self.vector_db_df is not None and not self.vector_db_df.empty:
                 self.logger.info("In-memory database built successfully from documents.")
            else:
                 self.logger.error("Failed to build index for in-memory database.")

        if self.vector_db_df is None or self.vector_db_df.empty:
             self.logger.critical("CRITICAL ERROR: In-memory Vector database could not be populated.")
             # raise ValueError("In-memory Vector database could not be populated.")


    def _build_index_and_populate_vector_db(self) -> Optional[pd.DataFrame]:
        """
        Builds the index from documents and populates the configured vector DB.
        If DB is CSV or InMemory, returns the DataFrame.
        If DB is ChromaDB, populates the collection directly.
        """
        self.logger.info(f"Starting index build process for vector DB type: {self.vector_db_type}")
        # --- 1. Document Parsing and Chunking ---
        all_chunks_info = self._parse_and_chunk_documents()
        if not all_chunks_info:
            self.logger.error("No chunks were extracted from the documents. Index building aborted.")
            return None

        # --- 2. Embedding Generation (Batched) ---
        chunk_texts = [info["chunk_text"] for info in all_chunks_info]
        self.logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        all_embeddings = self._batch_get_embeddings(chunk_texts, task_type="RETRIEVAL_DOCUMENT")

        # --- 3. Combine Info and Embeddings ---
        processed_chunks_data = []
        failed_embeddings = 0
        embedding_dim = None
        for i, chunk_info in enumerate(all_chunks_info):
            if i < len(all_embeddings) and all_embeddings[i] is not None:
                embedding = all_embeddings[i]
                if embedding_dim is None: embedding_dim = len(embedding)
                processed_chunks_data.append({**chunk_info, "embeddings": embedding})
            else:
                failed_embeddings += 1
                self.logger.warning(f"Failed to get embedding for chunk {i} from {chunk_info.get('document_name', 'N/A')}. Skipping.")

        if not processed_chunks_data:
            self.logger.error("No chunks were successfully embedded. Index building failed.")
            return None

        self.logger.info(f"Successfully embedded {len(processed_chunks_data)} chunks.")
        if failed_embeddings > 0:
             self.logger.warning(f"Failed to embed {failed_embeddings} chunks.")

        # --- 4. Populate Vector DB ---
        df = pd.DataFrame(processed_chunks_data)
        # Ensure embeddings are numpy arrays for CSV/InMemory
        df['embeddings'] = df['embeddings'].apply(np.array)

        if self.vector_db_type == "chroma":
            self.logger.info("Populating ChromaDB collection...")
            self._populate_chroma_from_dataframe(df)
            return None # ChromaDB handles its own storage
        else: # CSV or InMemory
            self.logger.info(f"Index data prepared for {self.vector_db_type} storage.")
            return df # Return the DataFrame

    def _parse_and_chunk_documents(self) -> List[Dict[str, Any]]:
        """Parses PDFs and splits them into chunks based on the configured strategy."""
        self.logger.info(f"Parsing documents using pattern: {self.documents_path_pattern}")
        document_files = glob.glob(self.documents_path_pattern)
        if not document_files:
            self.logger.error(f"No documents found matching pattern: {self.documents_path_pattern}")
            raise FileNotFoundError(f"No documents found matching pattern: {self.documents_path_pattern}")
        self.logger.info(f"Found {len(document_files)} documents to process.")

        if self.chunk_strategy == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\nḤadīth No.", "\n\n", "\n", ". ", ", ", " ", ""],
                length_function=len,
                is_separator_regex=False,
            )
        # Add other strategies here (e.g., semantic)
        # elif self.chunk_strategy == "semantic":
        #     self.logger.warning("Semantic chunking not yet fully implemented, using recursive.")
        #     # Placeholder for semantic chunking logic
        #     text_splitter = RecursiveCharacterTextSplitter(...) # Fallback or semantic implementation
        else:
             self.logger.warning(f"Unknown chunk strategy '{self.chunk_strategy}'. Defaulting to 'recursive'.")
             text_splitter = RecursiveCharacterTextSplitter(
                 chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                 separators=["\n\nḤadīth No.", "\n\n", "\n", ". ", ", ", " ", ""]
             )

        all_chunks_info = []
        total_chunks_processed = 0

        for doc_path in document_files:
            self.logger.debug(f"Processing document: {doc_path}")
            doc_name = os.path.basename(doc_path)
            try:
                with open(doc_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if not page_text or len(page_text.strip()) < 10: # Skip empty/very short pages
                                 self.logger.debug(f"Skipping empty/short page {page_num + 1} in {doc_name}")
                                 continue
                        except Exception as page_err:
                            self.logger.error(f"Error extracting text from {doc_name}, page {page_num + 1}: {page_err}")
                            continue

                        # Split text into chunks
                        try:
                             chunks = text_splitter.split_text(page_text)
                             for chunk_num, chunk_text in enumerate(chunks):
                                 cleaned_chunk = chunk_text.strip()
                                 if len(cleaned_chunk) >= 20: # Filter out very small chunks
                                     all_chunks_info.append({
                                         "document_name": doc_name,
                                         "page_number": page_num + 1,
                                         "chunk_number": chunk_num,
                                         "chunk_text": cleaned_chunk,
                                     })
                                     total_chunks_processed += 1
                                 else:
                                      self.logger.debug(f"Skipping short chunk {chunk_num} on page {page_num + 1} from {doc_name}")
                        except Exception as split_err:
                             self.logger.error(f"Error splitting text on page {page_num+1} of {doc_name}: {split_err}")

            except FileNotFoundError:
                self.logger.error(f"Document not found during processing: {doc_path}. Skipping.")
            except Exception as e:
                self.logger.exception(f"Error processing document {doc_path}: {e}")

        self.logger.info(f"Finished parsing. Extracted {len(all_chunks_info)} valid chunks from {len(document_files)} documents.")
        return all_chunks_info

    def _save_dataframe_to_csv(self, df: pd.DataFrame, file_path: str):
        """Saves the DataFrame (with embeddings as lists) to a CSV file."""
        try:
            self.logger.info(f"Saving vector database DataFrame to {file_path}...")
            df_copy = df.copy()
            # Convert numpy arrays to lists for CSV compatibility
            if 'embeddings' in df_copy.columns and isinstance(df_copy['embeddings'].iloc[0], np.ndarray):
                df_copy['embeddings'] = df_copy['embeddings'].apply(lambda x: x.tolist())
            df_copy.to_csv(file_path, index=False)
            self.logger.info(f"DataFrame successfully saved to {file_path}")
        except Exception as e:
            self.logger.exception(f"Error saving DataFrame to CSV at {file_path}: {e}")


    @retry(wait=wait_random_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
    def _get_embeddings_with_retry(self, text: str, task_type="RETRIEVAL_DOCUMENT") -> Optional[List[float]]:
        """Generates embeddings with retry logic and caching."""
        cache_key = f"{task_type}::{text}"
        if self.use_embedding_cache and cache_key in self.embedding_cache:
             self.logger.debug(f"Embedding cache HIT for task '{task_type}'")
             return self.embedding_cache[cache_key]

        self.logger.debug(f"Getting embedding for task '{task_type}'...")
        try:
            response = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type=task_type,
            )
            embedding = response['embedding']
            if self.use_embedding_cache:
                 self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            error_str = str(e).lower()
            if "resource has been exhausted" in error_str or "quota" in error_str or "429" in error_str:
                self.logger.warning(f"Rate limit or quota error during embedding: {e}. Retrying...")
                raise # Re-raise to trigger tenacity retry
            elif "api key not valid" in error_str or "permission denied" in error_str or "401" in error_str or "403" in error_str:
                 self.logger.error(f"Authentication/Permission Error during embedding: {e}. NOT RETRYING.")
                 return None # Don't retry auth errors
            elif "500" in error_str or "internal server error" in error_str:
                 self.logger.warning(f"Server error during embedding: {e}. Retrying...")
                 raise # Retry server errors
            else:
                self.logger.error(f"Non-retryable error generating embeddings: {e}")
                return None # Return None for other errors after retries fail

    def _batch_get_embeddings(self, text_list: List[str], batch_size=10, task_type="RETRIEVAL_DOCUMENT") -> List[Optional[List[float]]]:
        """
        Process embeddings for a list of texts.
        Note: Gemini API currently doesn't support true batching in one call.
        This function iterates and calls the embedding function for each text,
        applying rate limiting and handling potential failures.
        """
        self.logger.info(f"Getting embeddings for {len(text_list)} texts (simulated batching, size={batch_size})")
        all_embeddings: List[Optional[List[float]]] = []
        rate_limit_delay = 1.0 # Initial delay in seconds, adjust as needed

        for i, text in enumerate(text_list):
            self.logger.debug(f"Processing embedding for item {i+1}/{len(text_list)}")
            emb = None
            try:
                emb = self._get_embeddings_with_retry(text, task_type=task_type)
                if emb is None:
                    self.logger.warning(f"Embedding failed for item {i+1} after retries.")
                    # Optionally: Fallback to zero vector? Decide based on needs.
                    # emb = [0.0] * expected_dimension # Need to know expected dimension
            except Exception as e: # Catch final retry failure from tenacity
                 self.logger.error(f"Embedding failed permanently for item {i+1} after retries: {e}")
                 # emb remains None

            all_embeddings.append(emb)

            # Rate limiting: Sleep after each call
            # More sophisticated adaptive limiting could be added here based on 429 responses
            try:
                time.sleep(rate_limit_delay)
            except KeyboardInterrupt:
                 self.logger.warning("Embedding process interrupted.")
                 break

            # Log progress periodically
            if (i + 1) % batch_size == 0 or (i + 1) == len(text_list):
                successful_count = sum(1 for e in all_embeddings if e is not None)
                self.logger.info(f"Processed {i + 1}/{len(text_list)} embeddings ({successful_count} successful).")

        return all_embeddings

    def _get_relevant_chunks_csv(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves relevant chunks from the CSV/InMemory DataFrame."""
        if self.vector_db_df is None or self.vector_db_df.empty or 'embeddings' not in self.vector_db_df.columns:
            self.logger.warning("CSV/InMemory vector database is not available or lacks embeddings.")
            return []

        # Ensure query embedding is a 2D numpy array
        query_embedding_2d = np.array([query_embedding])

        # Check if embeddings column contains valid numpy arrays
        if not isinstance(self.vector_db_df['embeddings'].iloc[0], np.ndarray):
             self.logger.error("Embeddings in DataFrame are not numpy arrays. Cannot perform similarity search.")
             return []

        # Stack database embeddings (Potential memory issue for very large DBs)
        try:
             database_embeddings = np.stack(self.vector_db_df["embeddings"].to_numpy())
             if len(database_embeddings) == 0:
                  self.logger.warning("Database embeddings stack is empty.")
                  return []
             # Dimension check
             if query_embedding_2d.shape[1] != database_embeddings.shape[1]:
                  self.logger.error(f"Embedding dimension mismatch: Query ({query_embedding_2d.shape[1]}) vs DB ({database_embeddings.shape[1]})")
                  return []
        except Exception as stack_err:
             self.logger.error(f"Error stacking database embeddings: {stack_err}. Is the embedding data valid?")
             # Log a sample embedding for debugging
             if len(self.vector_db_df) > 0:
                  self.logger.debug(f"Sample embedding type: {type(self.vector_db_df['embeddings'].iloc[0])}, shape: {self.vector_db_df['embeddings'].iloc[0].shape if isinstance(self.vector_db_df['embeddings'].iloc[0], np.ndarray) else 'N/A'}")
             return []

        # Calculate cosine similarities
        try:
             similarities = cosine_similarity(query_embedding_2d, database_embeddings)[0]
        except Exception as sim_err:
             self.logger.error(f"Error calculating cosine similarity: {sim_err}")
             return []

        # Get top_k indices
        # Ensure top_k is not greater than the number of available chunks
        actual_top_k = min(top_k, len(similarities))
        if actual_top_k <= 0: return []

        # Get indices of top_k largest similarities
        top_indices = np.argsort(similarities)[-actual_top_k:][::-1]

        # Prepare results
        results = []
        relevant_rows = self.vector_db_df.iloc[top_indices]
        for i, (_, row) in enumerate(relevant_rows.iterrows()):
            similarity_score = similarities[top_indices[i]]
            results.append({
                "id": f"chunk_{top_indices[i]}", # Use index as ID for CSV
                "text": row['chunk_text'],
                "metadata": {
                    "document_name": row.get('document_name', 'Unknown'),
                    "page_number": int(row.get('page_number', -1)),
                    "chunk_number": int(row.get('chunk_number', -1)),
                },
                "score": float(similarity_score) # Ensure score is float
            })

        return results

    def _get_relevant_chunks_chroma(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves relevant chunks from ChromaDB."""
        if not hasattr(self, 'collection') or self.collection is None:
            self.logger.error("ChromaDB collection is not initialized.")
            return []
        try:
            # Query ChromaDB collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"] # Request distances
            )

            if not results or not results.get("ids") or not results["ids"][0]:
                self.logger.debug("No relevant chunks found in ChromaDB for the query.")
                return []

            # Format results consistently
            formatted_results = []
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert distance to similarity score (assuming cosine distance, range 0-2)
                # Similarity = 1 - (distance^2 / 2) if using squared L2
                # For cosine distance (often 1 - cosine_similarity): Similarity = 1 - distance
                similarity_score = 1.0 - distance # Adjust if Chroma uses a different distance metric internally
                formatted_results.append({
                    "id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": float(similarity_score) # Ensure score is float
                })

            return formatted_results

        except Exception as e:
            self.logger.exception(f"Error querying relevant chunks from ChromaDB: {e}")
            return []


    def _get_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant text chunks based on the query.
        Generates query embedding and calls the appropriate DB-specific retrieval method.
        Returns a list of dictionaries, each containing 'id', 'text', 'metadata', 'score'.
        """
        self.logger.info(f"Retrieving top {top_k} relevant chunks for query: '{query[:100]}...'")
        query_embedding = self._get_embeddings_with_retry(query, task_type="RETRIEVAL_QUERY")

        if query_embedding is None:
            self.logger.error("Query embedding failed. Cannot retrieve chunks.")
            return [] # Return empty list on failure

        results = []
        try:
            if self.vector_db_type == "chroma":
                results = self._get_relevant_chunks_chroma(query_embedding, top_k)
            elif self.vector_db_type in ["csv", "inmemory"]:
                 results = self._get_relevant_chunks_csv(query_embedding, top_k)
            else:
                 # Should not happen if __init__ validates type, but as a safeguard:
                 self.logger.error(f"Unsupported vector_db_type '{self.vector_db_type}' in _get_relevant_chunks.")
                 return []

            # Log retrieved chunk info (optional, can be verbose)
            if results:
                 self.logger.info(f"Retrieved {len(results)} chunks. Top score: {results[0]['score']:.4f}")
                 # for i, res in enumerate(results):
                 #     self.logger.debug(f"  Chunk {i+1}: ID={res['id']}, Score={res['score']:.4f}, Doc={res['metadata'].get('document_name')}, Page={res['metadata'].get('page_number')}")
            else:
                 self.logger.info("No relevant chunks retrieved.")

            return results

        except Exception as e:
             self.logger.exception(f"An unexpected error occurred in _get_relevant_chunks: {e}")
             return [] # Return empty list on unexpected errors

    def _format_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
         """Formats the retrieved chunks into a string for the LLM prompt."""
         if not chunks:
             return "No relevant context found."

         context_parts = []
         for chunk in chunks:
             metadata = chunk.get('metadata', {})
             score = chunk.get('score', 0.0)
             source_info = f"Source: Doc='{metadata.get('document_name', 'N/A')}', Page={metadata.get('page_number', 'N/A')}, Chunk={metadata.get('chunk_number', 'N/A')}, Score={score:.4f}"
             context_parts.append(f"[{source_info}]\n{chunk.get('text', '')}")

         return "\n\n---\n\n".join(context_parts)

    def _generate_llm_response(self, prompt: str) -> str:
        """Helper to call the generative model with caching and error handling."""
        if self.use_llm_cache and prompt in self.llm_cache:
            self.logger.debug("LLM cache HIT")
            return self.llm_cache[prompt]

        self.logger.debug(f"Generating LLM response for prompt (length: {len(prompt)}): '{prompt[:200]}...'")
        start_time = time.time()

        try:
            # Use the synchronous version
            response = self.model.generate_content(prompt)

            # Refined error/block checking for synchronous response
            if not response.candidates:
                 feedback = getattr(response, 'prompt_feedback', None)
                 block_reason = getattr(feedback, 'block_reason', 'Unknown')
                 safety_ratings = getattr(feedback, 'safety_ratings', [])
                 self.logger.warning(f"LLM Response was blocked or had no candidates. Reason: {block_reason}. Safety Ratings: {safety_ratings}")
                 # Return a specific error message indicating blockage
                 error_message = f"LLM_ERROR: Response blocked (Reason: {block_reason})."
                 # Cache the error message as well to avoid retrying blocked prompts
                 if self.use_llm_cache: self.llm_cache[prompt] = error_message
                 return error_message

            # Check if text attribute exists and is not empty
            generated_text = ""
            if hasattr(response, 'text') and response.text:
                 generated_text = response.text
            else:
                 # Fallback: Try to extract text from parts if 'text' is missing
                 try:
                     if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                          generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                          if not generated_text:
                              self.logger.warning("LLM Response generated but text content is empty even in parts.")
                              generated_text = "LLM_ERROR: Response content was empty."
                          else:
                               self.logger.debug("LLM Response text extracted from parts.")
                     else:
                          self.logger.warning("LLM Response generated but text attribute and parts are missing/empty.")
                          generated_text = "LLM_ERROR: Response structure unexpected or empty."
                 except (IndexError, AttributeError, Exception) as e:
                     self.logger.error(f"Error accessing response parts: {e}")
                     generated_text = "LLM_ERROR: Failed to access response content."

            elapsed_time = time.time() - start_time
            self.logger.debug(f"LLM generation took {elapsed_time:.2f}s. Response length: {len(generated_text)}")

            # Cache the result (even if it's an error message)
            if self.use_llm_cache:
                 self.llm_cache[prompt] = generated_text

            return generated_text

        except Exception as e:
            self.logger.exception(f"Exception during LLM generation call: {e}")
            error_str = str(e).lower()
            # Check for specific API errors
            if "resource_exhausted" in error_str or "quota" in error_str or "429" in error_str:
                return "LLM_ERROR: API quota limit reached."
            elif "api key not valid" in error_str or "permission denied" in error_str or "401" in error_str or "403" in error_str:
                 return "LLM_ERROR: Invalid API Key or insufficient permissions."
            # Handle other potential exceptions from the API client
            return f"LLM_ERROR: An unexpected error occurred during text generation: {str(e)}"

    # --- Query Processing Steps ---

    def _transform_query(self, query: str) -> str:
        """Transforms natural language numbers in the query to 'Hadith No. X' format."""
        query_lower = query.lower()
        # Simple replacements first
        replacements = {
            "first hadith": "Hadith No. 1", "hadith one": "Hadith No. 1", "hadith 1": "Hadith No. 1", "ḥadīth 1": "Hadith No. 1",
            "second hadith": "Hadith No. 2", "hadith two": "Hadith No. 2", "hadith 2": "Hadith No. 2", "ḥadīth 2": "Hadith No. 2",
            "third hadith": "Hadith No. 3", "hadith three": "Hadith No. 3", "hadith 3": "Hadith No. 3", "ḥadīth 3": "Hadith No. 3",
            "fourth hadith": "Hadith No. 4", "hadith four": "Hadith No. 4", "hadith 4": "Hadith No. 4", "ḥadīth 4": "Hadith No. 4",
            "fifth hadith": "Hadith No. 5", "hadith five": "Hadith No. 5", "hadith 5": "Hadith No. 5", "ḥadīth 5": "Hadith No. 5",
            "sixth hadith": "Hadith No. 6", "hadith six": "Hadith No. 6", "hadith 6": "Hadith No. 6", "ḥadīth 6": "Hadith No. 6",
            "seventh hadith": "Hadith No. 7", "hadith seven": "Hadith No. 7", "hadith 7": "Hadith No. 7", "ḥadīth 7": "Hadith No. 7",
            "eighth hadith": "Hadith No. 8", "hadith eight": "Hadith No. 8", "hadith 8": "Hadith No. 8", "ḥadīth 8": "Hadith No. 8",
            "ninth hadith": "Hadith No. 9", "hadith nine": "Hadith No. 9", "hadith 9": "Hadith No. 9", "ḥadīth 9": "Hadith No. 9",
            "tenth hadith": "Hadith No. 10", "hadith ten": "Hadith No. 10", "hadith 10": "Hadith No. 10", "ḥadīth 10": "Hadith No. 10",
            "eleventh hadith": "Hadith No. 11", "hadith eleven": "Hadith No. 11", "hadith 11": "Hadith No. 11", "ḥadīth 11": "Hadith No. 11",
            "twelfth hadith": "Hadith No. 12", "hadith twelve": "Hadith No. 12", "hadith 12": "Hadith No. 12", "ḥadīth 12": "Hadith No. 12",
            "thirteenth hadith": "Hadith No. 13", "hadith thirteen": "Hadith No. 13", "hadith 13": "Hadith No. 13", "ḥadīth 13": "Hadith No. 13",
            "fourteenth hadith": "Hadith No. 14", "hadith fourteen": "Hadith No. 14", "hadith 14": "Hadith No. 14", "ḥadīth 14": "Hadith No. 14",
            "fifteenth hadith": "Hadith No. 15", "hadith fifteen": "Hadith No. 15", "hadith 15": "Hadith No. 15", "ḥadīth 15": "Hadith No. 15",
            "sixteenth hadith": "Hadith No. 16", "hadith sixteen": "Hadith No. 16", "hadith 16": "Hadith No. 16", "ḥadīth 16": "Hadith No. 16",
            "seventeenth hadith": "Hadith No. 17", "hadith seventeen": "Hadith No. 17", "hadith 17": "Hadith No. 17", "ḥadīth 17": "Hadith No. 17",
            "eighteenth hadith": "Hadith No. 18", "hadith eighteen": "Hadith No. 18", "hadith 18": "Hadith No. 18", "ḥadīth 18": "Hadith No. 18",
            "nineteenth hadith": "Hadith No. 19", "hadith nineteen": "Hadith No. 19", "hadith 19": "Hadith No. 19", "ḥadīth 19": "Hadith No. 19",
            "twentieth hadith": "Hadith No. 20", "hadith twenty": "Hadith No. 20", "hadith 20": "Hadith No. 20", "ḥadīth 20": "Hadith No. 20",
            # ... (add 21-42 as needed, following the pattern) ...
            "twenty-first hadith": "Hadith No. 21", "hadith twenty-one": "Hadith No. 21", "hadith 21": "Hadith No. 21", "ḥadīth 21": "Hadith No. 21",
            "fortieth hadith": "Hadith No. 40", "hadith forty": "Hadith No. 40", "hadith 40": "Hadith No. 40", "ḥadīth 40": "Hadith No. 40",
            "forty-first hadith": "Hadith No. 41", "hadith forty-one": "Hadith No. 41", "hadith 41": "Hadith No. 41", "ḥadīth 41": "Hadith No. 41",
            "forty-second hadith": "Hadith No. 42", "hadith forty-two": "Hadith No. 42", "hadith 42": "Hadith No. 42", "ḥadīth 42": "Hadith No. 42",
        }
        transformed_query = query
        # Apply replacements - find longest match first to avoid partial replacements
        # Sort keys by length descending
        sorted_keys = sorted(replacements.keys(), key=len, reverse=True)
        found_match = False
        for key in sorted_keys:
            # Use regex to match whole words/phrases case-insensitively
            pattern = re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)
            if pattern.search(transformed_query): # Check if pattern exists before replacing
                 # Replace only the first occurrence found
                 transformed_query = pattern.sub(replacements[key], transformed_query, count=1)
                 self.logger.debug(f"Query transformed: '{key}' -> '{replacements[key]}'. Result: '{transformed_query}'")
                 found_match = True
                 break # Stop after first successful replacement

        # Regex for "Hadith number X" or "Hadith X"
        pattern_num = re.compile(r'hadith\s+(?:number\s+|#\s*)?(\d+)', re.IGNORECASE)
        match = pattern_num.search(transformed_query)
        if match and not found_match: # Apply only if specific word replacement didn't happen
            hadith_num = match.group(1)
            replacement_str = f"Hadith No. {hadith_num}"
            # Replace the found pattern with the standardized format
            transformed_query = pattern_num.sub(replacement_str, transformed_query, count=1)
            self.logger.debug(f"Query transformed using regex: Matched '{match.group(0)}' -> '{replacement_str}'. Result: '{transformed_query}'")

        if transformed_query != query:
            self.logger.info(f"Original Query: '{query}' -> Transformed Query: '{transformed_query}'")
        else:
             self.logger.debug(f"Query transformation did not alter the query: '{query}'")

        return transformed_query


    def _expand_query(self, query: str, num_expansions: int = 2) -> List[str]:
        """Generate query variations using an LLM to improve retrieval recall."""
        if num_expansions <= 0:
            return [query] # Return only original if no expansion needed

        self.logger.info(f"Expanding query '{query[:100]}...' into {num_expansions} variations.")

        expansion_prompt = f"""You are an expert search query generator specializing in Islamic texts, specifically Imam Nawawi's 40 Hadith.
        Generate {num_expansions} alternative search queries for the following original question.
        The goal is to capture the same information need using different phrasing, keywords, and potential concepts found in the Hadith collection.

        Original Question:
        "{query}"

        Instructions:
        1.  Understand the core intent of the original question.
        2.  Create {num_expansions} distinct alternative queries.
        3.  Use relevant Arabic terms (transliterated) if appropriate (e.g., 'niyyah', 'iman', 'islam', 'ihsan', 'halal', 'haram').
        4.  Focus on concepts, narrators (like 'Umar ibn al-Khattab', 'Abu Hurairah'), or Hadith themes if applicable.
        5.  Return ONLY the alternative queries, one per line. Do not include numbering, explanations, or the original query.

        Alternative Queries:"""

        response_text = self._generate_llm_response(expansion_prompt)

        if response_text.startswith("LLM_ERROR"):
             self.logger.warning(f"Query expansion failed due to LLM error: {response_text}")
             return [query] # Fallback to original query

        # Extract and clean expanded queries
        expanded_queries = [q.strip() for q in response_text.split('\n') if q.strip()]

        # Ensure the correct number of expansions, trimming or logging if needed
        if len(expanded_queries) > num_expansions:
             self.logger.debug(f"LLM generated more expansions than requested ({len(expanded_queries)}), taking first {num_expansions}.")
             expanded_queries = expanded_queries[:num_expansions]
        elif len(expanded_queries) < num_expansions:
             self.logger.debug(f"LLM generated fewer expansions than requested ({len(expanded_queries)}).")

        # Always include the original query transformed
        transformed_original = self._transform_query(query)
        all_queries = [transformed_original] + expanded_queries

        # Deduplicate the list of queries
        unique_queries = list(dict.fromkeys(all_queries))

        self.logger.info(f"Generated {len(unique_queries)} unique queries for retrieval: {unique_queries}")
        return unique_queries

    def _reason_and_refine_query(self, original_query: str, accumulated_context: str, history: List[str], current_hop: int) -> Tuple[str, str]:
        """
        Uses the LLM to analyze context and decide the next step for multihop.
        Improved parsing and stricter prompt instructions.

        Returns: Tuple[str, str]: (action_type, value). Action types: "ANSWER_COMPLETE", "NEXT_QUERY", "ERROR".
        """
        self.logger.info(f"--- Hop {current_hop} Reasoning ---")
        context_snippet = accumulated_context.strip()[:500] # Show snippet in log
        self.logger.debug(f"Reasoning based on History: {' -> '.join(history)}")
        self.logger.debug(f"Context Snippet Provided for Reasoning:\n{context_snippet if context_snippet else 'None'}...\n")

        # --- Stricter Prompt with Clearer Instructions ---
        prompt = f"""You are a precise reasoning engine for a RAG system answering questions about Imam Nawawi's 40 Hadith.
Your SOLE task is to determine the *next action* based on the Original Question and the Accumulated Context retrieved so far.

Original Question:
"{original_query}"

Accumulated Context (from previous search steps):
--- START CONTEXT ---
{accumulated_context if accumulated_context.strip() else "No context retrieved yet."}
--- END CONTEXT ---

Reasoning History (Previous search queries used):
--- START HISTORY ---
{' -> '.join(history) if history else "This is the first reasoning step."}
--- END HISTORY ---

Instructions:
1. Analyze the Original Question and the Accumulated Context. Focus on whether the context *directly and completely* answers the Original Question.
2. Consider the search queries used so far (Reasoning History).
3. Decide the single next action:
    a. If the Accumulated Context contains enough information to *fully answer* the Original Question: Output *exactly* this line and nothing else:
       ANSWER_COMPLETE
    b. If the Accumulated Context is insufficient or partially addresses the question, identify the *single most important missing piece of information* or the *next logical sub-question* needed to fully answer the Original Question. Formulate a *concise, specific search query* for this missing piece. Output *exactly* in this format (single line, nothing else):
       NEXT_QUERY: [Your concise search query here]
    c. If there was an error in retrieving context or the context is clearly irrelevant, and previous attempts haven't helped, output:
        ANSWER_COMPLETE

CRITICAL:
- Your entire response MUST be only ONE line.
- It must be EITHER "ANSWER_COMPLETE" OR "NEXT_QUERY: [your query]".
- Do NOT add explanations, introductions, apologies, or any other text.
- Do NOT answer the Original Question yourself. Only decide the next action.

Decision:"""

        response_text = self._generate_llm_response(prompt) # Use the generation helper

        # --- Improved Parsing with Regex ---
        action = "ERROR" # Default action if parsing fails
        value = f"LLM response parsing failed. Raw response: '{response_text}'" # Default value

        if response_text.startswith("LLM_ERROR"):
            self.logger.error(f"Reasoning step failed due to LLM error: {response_text}")
            value = response_text # Propagate LLM error
            # Decide whether to stop or assume complete on LLM error
            # Defaulting to ANSWER_COMPLETE to avoid loops on persistent LLM errors
            action = "ANSWER_COMPLETE"
            self.logger.warning("LLM error during reasoning. Assuming ANSWER_COMPLETE to proceed.")

        else:
             response_clean = response_text.strip()
             # Regex to capture the two allowed formats
             answer_complete_pattern = re.compile(r"^ANSWER_COMPLETE$", re.IGNORECASE)
             next_query_pattern = re.compile(r"^NEXT_QUERY:\s*(.+)$", re.IGNORECASE)

             match_complete = answer_complete_pattern.match(response_clean)
             match_next = next_query_pattern.match(response_clean)

             if match_complete:
                 action = "ANSWER_COMPLETE"
                 value = ""
                 self.logger.info(">>> Reasoning Decision: ANSWER_COMPLETE")
             elif match_next:
                 new_query = match_next.group(1).strip()
                 if new_query:
                     action = "NEXT_QUERY"
                     value = new_query
                     self.logger.info(f">>> Reasoning Decision: NEXT_QUERY -> '{new_query}'")
                 else:
                     # LLM generated NEXT_QUERY but the query was empty
                     action = "ANSWER_COMPLETE"
                     value = ""
                     self.logger.warning("LLM generated NEXT_QUERY but the query was empty. Treating as ANSWER_COMPLETE.")
             else:
                 # LLM response did not match expected formats
                 action = "ANSWER_COMPLETE" # Default to complete on format failure
                 value = ""
                 self.logger.warning(f"Unexpected reasoning response format: '{response_clean}'. LLM did not follow strict instructions. Defaulting to ANSWER_COMPLETE.")

        return action, value

    def _manage_context(self, accumulated_context_list: List[str], new_chunk_list: List[Dict[str, Any]],
                        max_tokens: int = 7000) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Manages the accumulated context list to stay within token limits.
        Includes deduplication and summarization of oldest context if needed.

        Args:
            accumulated_context_list: List of context strings from previous hops.
            new_chunk_list: List of new chunk dictionaries retrieved in the current hop.
            max_tokens: Approximate maximum token limit for the combined context.

        Returns:
            Tuple containing the updated list of context strings and the list of *unique* new chunks added.
        """
        if not new_chunk_list:
             self.logger.debug("No new chunks to manage.")
             return accumulated_context_list, []

        # --- Deduplication ---
        # 1. Deduplicate new chunks against themselves
        unique_new_chunks_by_text: Dict[str, Dict[str, Any]] = {}
        for chunk in new_chunk_list:
            text = chunk.get('text', '')
            if text not in unique_new_chunks_by_text:
                 unique_new_chunks_by_text[text] = chunk
            else:
                 # Keep the one with the higher score if duplicated
                 if chunk.get('score', 0) > unique_new_chunks_by_text[text].get('score', 0):
                      unique_new_chunks_by_text[text] = chunk
        deduped_new_chunks = list(unique_new_chunks_by_text.values())
        if len(deduped_new_chunks) < len(new_chunk_list):
             self.logger.debug(f"Removed {len(new_chunk_list) - len(deduped_new_chunks)} duplicate chunks from current hop.")

        # 2. Deduplicate new chunks against existing context (simple string check)
        # More sophisticated semantic deduplication could be added here.
        existing_texts = set()
        for ctx_str in accumulated_context_list:
             # Extract text content, handling potential formatting like "[Source: ...] Text"
             match = re.search(r'\]:\s*(.*)', ctx_str, re.DOTALL)
             text_content = match.group(1).strip() if match else ctx_str.strip()
             existing_texts.add(text_content) # Might need refinement based on actual context format

        truly_new_chunks = []
        for chunk in deduped_new_chunks:
             text = chunk.get('text', '')
             # Use SequenceMatcher for similarity check (adjust threshold as needed)
             is_similar = False
             for existing_text in existing_texts:
                   similarity = SequenceMatcher(None, text, existing_text).ratio()
                   if similarity > 0.9: # High threshold for near-duplicates
                       is_similar = True
                       self.logger.debug(f"Found new chunk similar to existing context (Score: {similarity:.2f}). Skipping.")
                       break
             if not is_similar:
                  truly_new_chunks.append(chunk)
                  existing_texts.add(text) # Add its text to avoid adding another similar one later

        if len(truly_new_chunks) < len(deduped_new_chunks):
             self.logger.info(f"Removed {len(deduped_new_chunks) - len(truly_new_chunks)} chunks similar to existing context.")

        if not truly_new_chunks:
            self.logger.info("No truly new, unique context chunks found in this hop.")
            return accumulated_context_list, []

        # Format the truly new chunks into a single context string for this hop
        new_context_string = self._format_context_from_chunks(truly_new_chunks)
        current_context_list = accumulated_context_list + [new_context_string]

        # --- Size Management (Summarization) ---
        # Rough token count estimation (3-4 chars ≈ 1 token, use 3.5 for safety)
        total_chars = sum(len(ctx) for ctx in current_context_list)
        estimated_tokens = total_chars / 3.5

        self.logger.debug(f"Context size before potential summarization: ~{estimated_tokens:.0f} tokens")

        # Iteratively summarize oldest context if over limit
        while estimated_tokens > max_tokens and len(current_context_list) > 1:
            self.logger.info(f"Context too large ({estimated_tokens:.0f} > {max_tokens} tokens), summarizing oldest content...")

            oldest_context = current_context_list.pop(0) # Remove oldest

            # Check if it's already summarized
            if oldest_context.strip().startswith("[SUMMARIZED CONTEXT]:"):
                 self.logger.warning("Oldest context is already summarized. Cannot summarize further. Keeping it.")
                 current_context_list.insert(0, oldest_context) # Put it back if already summarized
                 break # Avoid infinite loop if only summarized context remains

            summary_prompt = f"""Summarize the following context section from Imam Nawawi's 40 Hadith collection.
            Preserve all key facts, Hadith references (like 'Hadith No. X'), narrator names, core teachings, and specific details mentioned.
            Be concise but retain the essential information needed to answer questions based on this section.

            Context Section to Summarize:
            ---
            {oldest_context}
            ---

            Concise Summary:"""

            summarized = self._generate_llm_response(summary_prompt)

            if summarized.startswith("LLM_ERROR"):
                 self.logger.error(f"Failed to summarize context: {summarized}. Keeping original.")
                 # Put the original back if summarization failed
                 current_context_list.insert(0, oldest_context)
                 # Break loop to avoid retrying summarization if LLM is failing
                 break
            else:
                # Add the summary to the beginning of the list
                current_context_list.insert(0, f"[SUMMARIZED CONTEXT]:\n{summarized}")
                self.logger.info("Summarization complete. Added summarized context.")

            # Recalculate token count
            total_chars = sum(len(ctx) for ctx in current_context_list)
            estimated_tokens = total_chars / 3.5
            self.logger.info(f"Context size after summarization: ~{estimated_tokens:.0f} tokens")

        if estimated_tokens > max_tokens:
             self.logger.warning(f"Context size (~{estimated_tokens:.0f}) still exceeds max_tokens ({max_tokens}) even after attempting summarization.")

        return current_context_list, truly_new_chunks # Return updated list and the new unique chunks added


    def _generate_final_answer(self, original_query: str, accumulated_context_list: List[str]) -> str:
        """
        Generates the final answer using the LLM, specifically prompted for multihop context synthesis.
        """
        self.logger.info("Generating final answer from accumulated context...")

        # Combine context from all hops/summaries for the final prompt
        # Use a very distinct separator
        final_context_str = "\n\n==== CONTEXT FROM HOP/SUMMARY SEPARATOR ====\n\n".join(accumulated_context_list)

        if not final_context_str or final_context_str.strip() == "No relevant context found.":
             self.logger.warning("No context available for final answer generation.")
             return "Based on the information retrieved, I could not find a specific answer to your question."

        # --- Prompt explicitly designed for Multihop Synthesis ---
        prompt = f"""You are an expert assistant knowledgeable about Imam Nawawi's 40 Hadith.
Your task is to synthesize a final, comprehensive answer to the user's *Original Question* using *only* the information found within the *Accumulated Context* provided below. The context comes from potentially multiple search steps and may include summaries.

Instructions:
1.  Carefully read the *Original Question*.
2.  Carefully read *all sections* within the Accumulated Context, noting any summaries indicated by "[SUMMARIZED CONTEXT]:".
3.  Synthesize a single, coherent answer that directly and fully addresses the *Original Question*. Combine relevant pieces of information accurately.
4.  Base your answer *strictly* on the text present in the Accumulated Context. Do *not* add external information, interpretations, or opinions not explicitly stated. Do not make assumptions or guess.
5.  If the Accumulated Context *definitively lacks* the information needed to fully answer the Original Question (after reviewing all sections), state clearly: "Based on the provided context, the specific information about [missing aspect] could not be found." Do not apologize excessively.
6.  Structure the answer clearly and logically. Start directly with the answer.
7.  Do NOT refer to the search process (e.g., "Based on the retrieved context..."). Just provide the synthesized answer based *on* the context.
8.  Do NOT include the source markers (like `[Source: ...]`) in your final answer.

Original Question:
"{original_query}"

Accumulated Context:
--- START CONTEXT ---
{final_context_str}
--- END CONTEXT ---

Final Synthesized Answer:"""

        final_response = self._generate_llm_response(prompt)

        # Check if the LLM itself failed
        if final_response.startswith("LLM_ERROR"):
            self.logger.error(f"Final answer generation failed due to LLM error: {final_response}")
            # Provide a user-friendly error message
            return f"I encountered an error while trying to generate the final answer ({final_response}). Please try again."
        else:
             self.logger.info(f"Final answer generated (length: {len(final_response)}). Snippet: {final_response[:200]}...")
             return final_response

    # --- Evaluation and Feedback ---

    def evaluate_response(self, question: str, answer: str, retrieved_context_str: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of the generated answer using LLM-based metrics.
        Optionally compares against ground truth if provided.
        """
        self.logger.info("Evaluating generated response...")
        metrics = {}

        # 1. Relevance Score (Answer vs. Question)
        try:
            relevance_prompt = f"""Rate the relevance of the Answer to the Question on a scale of 1 (Not relevant) to 5 (Highly relevant).
            Consider if the answer directly addresses the core intent of the question.

            Question: "{question}"
            Answer: "{answer}"

            Output format (JSON): {{"relevance_rating": [1-5], "relevance_explanation": "[Brief explanation]"}}
            """
            relevance_response = self._generate_llm_response(relevance_prompt)
            match = re.search(r'{\s*"relevance_rating"\s*:\s*(\d+)\s*,\s*"relevance_explanation"\s*:\s*"([^"]+)"\s*}', relevance_response, re.IGNORECASE | re.DOTALL)
            if match:
                metrics['relevance_rating'] = int(match.group(1))
                metrics['relevance_explanation'] = match.group(2).strip()
            else:
                 self.logger.warning(f"Could not parse relevance rating from response: {relevance_response}")
                 metrics['relevance_rating'] = None
                 metrics['relevance_explanation'] = f"Parsing Failed: {relevance_response}"
        except Exception as e:
            self.logger.error(f"Error during relevance evaluation: {e}")
            metrics['relevance_rating'] = None
            metrics['relevance_explanation'] = f"Error: {e}"

        # 2. Faithfulness Score (Answer vs. Context)
        try:
            faithfulness_prompt = f"""Rate how faithful the Answer is to the provided Context on a scale of 1 (Contradicts context) to 5 (Fully supported by context).
            The answer should only contain information present in the context.

            Context:
            ---
            {retrieved_context_str[:6000]}... # Truncate context if too long for eval prompt
            ---
            Answer: "{answer}"

            Output format (JSON): {{"faithfulness_rating": [1-5], "faithfulness_explanation": "[Brief explanation highlighting support or contradiction]"}}
            """
            faithfulness_response = self._generate_llm_response(faithfulness_prompt)
            match = re.search(r'{\s*"faithfulness_rating"\s*:\s*(\d+)\s*,\s*"faithfulness_explanation"\s*:\s*"([^"]+)"\s*}', faithfulness_response, re.IGNORECASE | re.DOTALL)
            if match:
                metrics['faithfulness_rating'] = int(match.group(1))
                metrics['faithfulness_explanation'] = match.group(2).strip()
            else:
                 self.logger.warning(f"Could not parse faithfulness rating from response: {faithfulness_response}")
                 metrics['faithfulness_rating'] = None
                 metrics['faithfulness_explanation'] = f"Parsing Failed: {faithfulness_response}"
        except Exception as e:
            self.logger.error(f"Error during faithfulness evaluation: {e}")
            metrics['faithfulness_rating'] = None
            metrics['faithfulness_explanation'] = f"Error: {e}"

        # 3. Ground Truth Comparison (if available) - Basic Similarity or LLM Eval
        if ground_truth:
            try:
                gt_prompt = f"""Compare the Generated Answer to the Ground Truth Answer for the given Question. Rate the similarity/correctness on a scale of 1 (Very different/Incorrect) to 5 (Highly similar/Correct).

                Question: "{question}"
                Ground Truth Answer: "{ground_truth}"
                Generated Answer: "{answer}"

                Output format (JSON): {{"ground_truth_similarity_rating": [1-5], "ground_truth_similarity_explanation": "[Brief explanation of similarity/differences]"}}
                """
                gt_response = self._generate_llm_response(gt_prompt)
                match = re.search(r'{\s*"ground_truth_similarity_rating"\s*:\s*(\d+)\s*,\s*"ground_truth_similarity_explanation"\s*:\s*"([^"]+)"\s*}', gt_response, re.IGNORECASE | re.DOTALL)
                if match:
                     metrics['ground_truth_similarity_rating'] = int(match.group(1))
                     metrics['ground_truth_similarity_explanation'] = match.group(2).strip()
                else:
                     self.logger.warning(f"Could not parse ground truth similarity rating from response: {gt_response}")
                     metrics['ground_truth_similarity_rating'] = None
                     metrics['ground_truth_similarity_explanation'] = f"Parsing Failed: {gt_response}"
            except Exception as e:
                self.logger.error(f"Error during ground truth comparison: {e}")
                metrics['ground_truth_similarity_rating'] = None
                metrics['ground_truth_similarity_explanation'] = f"Error: {e}"


        self.logger.info(f"Evaluation Metrics: {metrics}")
        return metrics

    def _assess_answer_confidence(self, answer: str, context: str, query: str) -> Optional[float]:
        """Estimate confidence (0.0 to 1.0) in the answer based on context support."""
        self.logger.debug("Assessing answer confidence...")
        prompt = f"""Assess how well the following Answer is supported *solely* by the provided Context in relation to the Query.
        Provide a confidence score between 0.0 (no support) and 1.0 (fully supported).
        Consider only the information explicitly present in the Context.

        Query: {query}
        Context:
        ---
        {context[:6000]}... # Truncate context
        ---
        Answer: {answer}

        Confidence Score (Return ONLY the numerical score, e.g., 0.85):"""

        response = self._generate_llm_response(prompt)

        if response.startswith("LLM_ERROR"):
            self.logger.warning(f"Confidence assessment failed due to LLM error: {response}")
            return None

        try:
            # Extract float, handling potential surrounding text
            match = re.search(r"(\d\.\d+)", response)
            if match:
                confidence = float(match.group(1))
                confidence = max(0.0, min(1.0, confidence)) # Clamp between 0 and 1
                self.logger.debug(f"Assessed Confidence Score: {confidence:.2f}")
                return confidence
            else:
                self.logger.warning(f"Could not parse confidence score from response: '{response}'. Returning moderate confidence.")
                return 0.5 # Default moderate confidence if parsing fails
        except ValueError:
            self.logger.warning(f"Could not convert confidence score response to float: '{response}'. Returning moderate confidence.")
            return 0.5
        except Exception as e:
             self.logger.error(f"Error parsing confidence score: {e}")
             return None


    def _load_feedback_history(self):
        """Load feedback history from CSV if it exists."""
        if os.path.exists(self.feedback_db_path):
            try:
                feedback_df = pd.read_csv(self.feedback_db_path)
                # Convert relevant columns back to appropriate types if needed
                # Example: feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
                self.feedback_history = feedback_df.to_dict('records')
                self.logger.info(f"Loaded {len(self.feedback_history)} feedback records from {self.feedback_db_path}.")
            except Exception as e:
                self.logger.warning(f"Could not load feedback history from {self.feedback_db_path}: {e}")
                self.feedback_history = []
        else:
            self.logger.info(f"Feedback history file not found at {self.feedback_db_path}. Starting fresh.")
            self.feedback_history = []

    def _save_feedback_history(self):
        """Save the current feedback history to CSV."""
        if not self.feedback_history:
            self.logger.debug("No feedback history to save.")
            return

        try:
            feedback_df = pd.DataFrame(self.feedback_history)
            feedback_df.to_csv(self.feedback_db_path, index=False)
            self.logger.info(f"Saved {len(self.feedback_history)} feedback records to {self.feedback_db_path}.")
        except Exception as e:
            self.logger.error(f"Failed to save feedback history to {self.feedback_db_path}: {e}")

    def record_feedback(self, query: str, answer: str, hop_count: int,
                        final_context: str, reasoning_trace: List[str],
                        user_rating: Optional[int] = None, # e.g., 1-5 stars
                        user_comment: Optional[str] = None,
                        evaluation_metrics: Optional[Dict[str, Any]] = None):
        """
        Record user feedback and internal metrics for a query-answer interaction.

        Args:
            query: The original user query.
            answer: The final generated answer.
            hop_count: Number of hops taken.
            final_context: The final accumulated context string used for the answer.
            reasoning_trace: The step-by-step reasoning log.
            user_rating: Optional user-provided rating (e.g., 1-5).
            user_comment: Optional user textual feedback.
            evaluation_metrics: Optional dictionary of automated evaluation scores.
        """
        feedback_record = {
            "timestamp": pd.Timestamp.now().isoformat(), # Use ISO format string
            "query": query,
            "answer": answer,
            "hop_count": hop_count,
            "final_context_length": len(final_context),
            # Avoid storing very long context/traces directly in CSV, maybe store hash or summary
            # "final_context": final_context[:1000] + "...", # Store truncated context
            "reasoning_trace": " -> ".join(reasoning_trace), # Store trace as string
            "user_rating": user_rating,
            "user_comment": user_comment,
            **(evaluation_metrics or {}) # Add automated metrics
        }

        self.feedback_history.append(feedback_record)
        self.logger.info(f"Recorded feedback for query: '{query[:50]}...' (User Rating: {user_rating})")

        # Save feedback periodically or immediately
        self._save_feedback_history()

        # Optional: Trigger analysis if conditions met
        # if user_rating is not None and user_rating <= 2 and len(self.feedback_history) >= 10:
        #      self._analyze_feedback_patterns()


    def _analyze_feedback_patterns(self):
        """Placeholder for analyzing feedback to identify improvement areas."""
        self.logger.info("Analyzing feedback patterns (Placeholder)...")
        # TODO: Implement logic to analyze low-rated responses, common issues, etc.
        # Example: Find common themes in low-rated queries or comments.
        # Example: Correlate low ratings with specific reasoning steps or context issues.
        if not self.feedback_history: return

        low_rated = [f for f in self.feedback_history if f.get('user_rating') is not None and f['user_rating'] <= 2]
        if low_rated:
             self.logger.info(f"Found {len(low_rated)} low-rated feedback entries.")
             # Analyze common queries, comments, evaluation scores in low_rated entries...


    # --- Main Query Processing ---

    def process_query(self, question: str, top_k: int = 5, max_hops: int = 3,
                      use_query_expansion: bool = False,
                      request_evaluation: bool = True,
                      generate_graph: bool = True, # New flag
                      record_user_feedback: Optional[Dict[str, Any]] = None
                      ) -> Dict[str, Any]:
        """
        Processes a user query using the enhanced multihop RAG pipeline.
        Includes options for graph generation and formatted reasoning.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve per hop.
            max_hops: Maximum number of retrieval/reasoning hops.
            use_query_expansion: If True, expands the initial query.
            request_evaluation: If True, runs automated evaluation on the final answer.
            record_user_feedback: Optional dict containing 'rating' and/or 'comment'.

        Returns:
            A dictionary containing:
            - 'final_answer': The generated answer string.
            - 'source_info': Description of the process.
            - 'reasoning_trace': List of strings detailing the multi-hop process.
            - 'confidence_score': Estimated confidence (0.0-1.0) or None.
            - 'evaluation_metrics': Dictionary of automated metrics (if requested) or None.
            - 'debug_info': Dictionary with timing, hop count, context length.
        """
        process_start_time = time.time()
        self.logger.info(f"\n--- Starting Query Process ---")
        self.logger.info(f"Original Question: '{question}'")
        self.logger.info(f"Config: top_k={top_k}, max_hops={max_hops}, expansion={use_query_expansion}, eval={request_evaluation}")

        # --- Initialization ---
        # Basic check for vector DB availability
        db_ready = False
        if self.vector_db_type == 'chroma' and hasattr(self, 'collection') and self.collection is not None and self.collection.count() > 0:
            db_ready = True
        elif self.vector_db_type in ['csv', 'inmemory'] and self.vector_db_df is not None and not self.vector_db_df.empty:
             db_ready = True

        if not db_ready:
            self.logger.critical("Vector database is not initialized or empty. Cannot process query.")
            return {
                "final_answer": "Error: The knowledge base (Vector DB) is not available. Please check configuration and data.",
                "source_info": "Initialization Error",
                "reasoning_trace": ["Vector DB not ready"],
                "confidence_score": None,
                "evaluation_metrics": None,
                "debug_info": {"processing_time_s": time.time() - process_start_time, "hops_taken": 0, "final_context_length": 0}
            }

        original_query = question
        # Initial query transformation
        current_query = self._transform_query(original_query)
        # search_query_history = [current_query] # Start with the initial transformed query

        # Optional Query Expansion for the first hop
        queries_to_try = [current_query]
        if use_query_expansion:
            self.logger.info("Applying query expansion for the initial retrieval...")
            queries_to_try = self._expand_query(current_query, num_expansions=2) # Get 1 original + 2 expansions


        accumulated_context_list = [] # Stores context strings from each hop/summary
        reasoning_trace = ["START"] # Track the LLM's reasoning steps
        search_query_history = [current_query] # Start with the initial transformed query
        hops_taken = 0
        all_retrieved_chunks_this_session = [] # Keep track of all unique chunks retrieved


        # --- Multihop Loop ---
        for hop in range(max_hops):
            hops_taken = hop + 1
            self.logger.info(f"\n--- Hop {hops_taken}/{max_hops} ---")
            reasoning_trace.append(f"--- Hop {hops_taken} ---")

            # --- Retrieval ---
            # If using expansion, might retrieve for multiple queries here, then merge/rank results.
            # Simple approach for now: Use the single 'current_query' determined by reasoning.
            # If it's the first hop AND expansion is enabled, retrieve for all expanded queries?
            # Let's stick to the single current_query for simplicity in the loop.
            # Expansion primarily helps the *first* query generation or reasoning.

            if not current_query: # Should not happen if reasoning works, but safety check
                 self.logger.warning("Current query is empty. Stopping hop.")
                 reasoning_trace.append(f"Hop {hops_taken}: Error - Query is empty. Breaking loop.")
                 break

            self.logger.info(f"Retrieving context for query: '{current_query}'")
            search_query_history.append(current_query)
            reasoning_trace.append(f"Hop {hops_taken}: Retrieving with query -> '{current_query}'")

            retrieved_chunks = self._get_relevant_chunks(current_query, top_k=top_k)

            if not retrieved_chunks:
                self.logger.warning(f"Retrieval returned no relevant chunks for hop {hops_taken}.")
                reasoning_trace.append(f"Hop {hops_taken}: Retrieval yielded no results.")
                # If first hop fails completely, maybe stop early?
                if hop == 0:
                    self.logger.warning("Initial retrieval failed. Aborting multi-hop.")
                    reasoning_trace.append("Hop 1: Initial retrieval failed. Stopping.")
                    break # Exit loop early if first hop yields nothing
                # Otherwise, let reasoning decide based on existing context (if any)
            else:
                 self.logger.info(f"Retrieved {len(retrieved_chunks)} chunks for hop {hops_taken}.")
                 reasoning_trace.append(f"Hop {hops_taken}: Retrieved {len(retrieved_chunks)} chunk(s). Top score: {retrieved_chunks[0]['score']:.4f}")
                 all_retrieved_chunks_this_session.extend(retrieved_chunks) # Add to session total


            # --- Context Management (Deduplication & Size Control) ---
            # Pass the currently accumulated list and the newly retrieved chunks
            accumulated_context_list, added_unique_chunks = self._manage_context(
                 accumulated_context_list,
                 retrieved_chunks, # Pass the list of dicts
                 max_tokens=7000 # Leave buffer for prompt instructions, query, etc.
            )
            if added_unique_chunks:
                 reasoning_trace.append(f"Hop {hops_taken}: Added {len(added_unique_chunks)} unique chunks to context.")
            else:
                 reasoning_trace.append(f"Hop {hops_taken}: No new unique chunks added to context.")


            # Combine all context gathered *so far* for the reasoning step
            # Use a distinct separator for the reasoning prompt
            full_accumulated_text_for_reasoning = "\n\n==== CONTEXT FROM HOP/SUMMARY SEPARATOR ====\n\n".join(accumulated_context_list)

            # --- Reasoning ---
            action, value = self._reason_and_refine_query(
                original_query,
                full_accumulated_text_for_reasoning,
                search_query_history, # Pass history of search terms used
                hops_taken
            )
            reasoning_trace.append(f"Hop {hops_taken}: Reasoning result -> Action='{action}', Value='{value[:100]}...'")

            # --- Process Reasoning Outcome ---
            if action == "ANSWER_COMPLETE":
                self.logger.info("Reasoning concluded: Sufficient context likely found or cannot improve.")
                reasoning_trace.append("Hop {hops_taken}: Reasoning -> ANSWER_COMPLETE. Proceeding to final answer.")
                break # Exit the loop to proceed to final generation
            elif action == "NEXT_QUERY":
                # Update query for the *next* iteration, apply transformations
                next_raw_query = value
                current_query = self._transform_query(next_raw_query)
                reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> NEXT_QUERY = '{current_query}'")
                # Loop continues...
            elif action == "ERROR":
                 self.logger.error(f"Error during reasoning step: {value}. Stopping multihop process.")
                 reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> ERROR: {value}. Proceeding to final answer with current context.")
                 break # Exit loop and proceed to final generation with current context
            else: # Unexpected action
                 self.logger.warning(f"Unexpected action '{action}' from reasoning step. Stopping.")
                 reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> UNEXPECTED ACTION '{action}'. Proceeding to final answer.")
                 break # Exit loop

        # --- End of Loop ---
        if hops_taken == max_hops and action != "ANSWER_COMPLETE":
             self.logger.info("Max hops reached. Proceeding to final answer generation.")
             reasoning_trace.append(f"Max hops ({max_hops}) reached.")

        reasoning_trace.append("--- Final Answer Generation ---")

        # --- Final Answer Generation ---
        final_answer = self._generate_final_answer(original_query, accumulated_context_list)


        # --- Confidence & Evaluation ---
        confidence_score = None
        evaluation_metrics = None
        final_context_str = "\n\n==== CONTEXT FROM HOP/SUMMARY SEPARATOR ====\n\n".join(accumulated_context_list)

        if not final_answer.startswith("LLM_ERROR:") and final_context_str:
            confidence_score = self._assess_answer_confidence(final_answer, final_context_str, original_query)
            reasoning_trace.append(f"Confidence Score: {confidence_score:.2f}" if confidence_score is not None else "Confidence Score: N/A")

            if request_evaluation:
                 evaluation_metrics = self.evaluate_response(original_query, final_answer, final_context_str)
                 reasoning_trace.append(f"Evaluation Metrics: {evaluation_metrics}")

        formatted_reasoning = self._format_reasoning_trace(reasoning_trace)
        
        # --- NEW: Generate Graph ---
        graph_filename = None
        if generate_graph:
            # Pass the actual queries used for retrieval (search_query_history)
            graph_filename = self._generate_hop_graph(reasoning_trace, search_query_history)


        # --- Wrap Up ---
        process_end_time = time.time()
        processing_time = process_end_time - process_start_time
        self.logger.info(f"--- Query Process Finished ---")
        self.logger.info(f"Total Processing time: {processing_time:.2f} seconds")
        self.logger.info(f"Final Answer Snippet: {final_answer[:200]}...")

        debug_info = {
             "processing_time_s": round(processing_time, 2),
             "hops_taken": hops_taken,
             "queries_used": search_query_history,
             "final_context_length": len(final_context_str),
             "vector_db_type": self.vector_db_type,
             "cache_hits_embedding": sum(1 for t in reasoning_trace if "Embedding cache HIT" in t), # Simple count
             "cache_hits_llm": sum(1 for t in reasoning_trace if "LLM cache HIT" in t), # Simple count
             "graph_filename": graph_filename # Add graph filename here
        }

        # --- Record Feedback ---
        user_rating = record_user_feedback.get('rating') if record_user_feedback else None
        user_comment = record_user_feedback.get('comment') if record_user_feedback else None
        self.record_feedback(
            query=original_query,
            answer=final_answer,
            hop_count=hops_taken,
            final_context=final_context_str,
            reasoning_trace=reasoning_trace,
            user_rating=user_rating,
            user_comment=user_comment,
            evaluation_metrics=evaluation_metrics
        )

        return {
            "final_answer": final_answer,
            "source_info": f"Synthesized from context retrieved over {hops_taken} hop(s) using {self.vector_db_type} DB.",
            "reasoning_trace": reasoning_trace, # Still return raw trace if needed
            "formatted_reasoning": formatted_reasoning, # Add formatted trace
            "confidence_score": confidence_score,
            "evaluation_metrics": evaluation_metrics,
            "debug_info": debug_info # Contains graph filename now
        }

# --- End of HadidRAG Class ---

# Example Usage (Illustrative)
# if __name__ == "__main__":
#     print("Example RAG Usage...")

#     # Ensure documents path is correct or adjust DOCUMENTS_PATH_PATTERN
#     # Make sure 'Imam_Nawawi' directory exists relative to the script
#     # and contains the PDF file(s).
#     doc_pattern = "Imam_Nawawi/*.pdf" # Or specify the exact PDF path
#     if not glob.glob(doc_pattern):
#          print(f"\nWARNING: No documents found at '{doc_pattern}'.")
#          print("Please ensure the path is correct and PDF files are present.")
#          # Example: Create a dummy file for testing if needed (won't have real content)
#          # DUMMY_DIR = "Imam_Nawawi"
#          # DUMMY_FILE = os.path.join(DUMMY_DIR, "dummy_hadith.txt")
#          # if not os.path.exists(DUMMY_DIR): os.makedirs(DUMMY_DIR)
#          # if not os.path.exists(DUMMY_FILE):
#          #     with open(DUMMY_FILE, "w") as f: f.write("Hadith No. 1: Actions are by intentions.")
#          #     doc_pattern = os.path.join(DUMMY_DIR, "*.txt") # Adjust pattern if using dummy txt
#          #     print(f"Using dummy text file pattern: {doc_pattern}")
#          # exit() # Exit if no real docs and no dummy handling


#     try:
#         # Initialize with desired settings
#         rag = HadidRAG(
#             vector_db_type="chroma", # Try "chroma" or "csv" or "inmemory"
#             log_level="INFO", # Use "DEBUG" for more verbose logs
#             documents_path=doc_pattern,
#             # vector_db_path="my_custom_hadith_db.csv" # Optional: Custom path
#         )

#         # --- Test Queries ---
#         queries = [
#             "What is the main theme of the first hadith?",
#             "Explain the concept of Ihsan based on Hadith number 2.", # Transformed to Hadith No. 2
#             "Who narrated the hadith about the pillars of Islam?", # Requires multi-hop potentially
#             "Summarize Hadith 18 on Taqwa.",
#             "What actions are forbidden in Hadith 24?",
#             "Tell me about sincerity (ikhlas) mentioned in the hadith."
#         ]

#         for i, query in enumerate(queries):
#             print(f"\n--- Processing Query {i+1}: {query} ---")
#             result = rag.process_query(
#                 question=query,
#                 top_k=5,
#                 max_hops=3,
#                 request_evaluation=True, # Request evaluation
#                 # record_user_feedback={'rating': 5, 'comment': 'Looks good!'} # Simulate feedback
#             )

#             print(f"\nFinal Answer:\n{result['final_answer']}")
#             print(f"\nSource Info: {result['source_info']}")
#             print(f"Confidence Score: {result['confidence_score']:.2f}" if result['confidence_score'] is not None else "Confidence Score: N/A")
#             # print(f"\nReasoning Trace:")
#             # for step in result['reasoning_trace']:
#             #     print(f"- {step}")
#             print(f"\nEvaluation Metrics: {result['evaluation_metrics']}")
#             print(f"Debug Info: {result['debug_info']}")
#             print("-" * 50)
#             time.sleep(2) # Small delay between queries if needed

#     except FileNotFoundError as e:
#          print(f"\nERROR: {e}")
#          print("Please ensure the document path pattern in the script or constructor is correct.")
#     except ValueError as e:
#          print(f"\nConfiguration Error: {e}")
#     except Exception as e:
#         print(f"\nAn unexpected error occurred during execution: {e}")
#         import traceback
#         traceback.print_exc()