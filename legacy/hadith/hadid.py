import os
import glob
import re
import PyPDF2
import numpy as np
import pandas as pd
from typing import Any, Tuple, List
import google.generativeai as genai
from google.generativeai import types as genai_types # Alias to avoid conflict
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from dotenv import load_dotenv
load_dotenv()

# No asyncio or nest_asyncio needed here anymore

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=API_KEY)

MODEL_ID = "gemini-2.0-flash"
TEXT_EMBEDDING_MODEL = "text-embedding-004"
VECTOR_DB_PATH = "vector_db_mini_vertex.csv"
DOCUMENTS_PATH_PATTERN = "Imam_Nawawi/*.pdf"

class HadidRAG:
    """
    Encapsulates the RAG logic for Imam Nawawi's 40 Hadith.
    Handles resource loading and query processing synchronously.
    """
    def __init__(self):
        print("Initializing HadidRAG...")
        self._initialize_clients()
        self._load_or_build_vector_db()
        print("HadidRAG Initialized Successfully.")

    def _initialize_clients(self):
        """Initializes Google AI clients."""
        try:
            self.embedding_model_name = f"models/{TEXT_EMBEDDING_MODEL}"
            generation_config = {
                "temperature": 0.8, "top_p": 0.95, "top_k": 40,
                "max_output_tokens": 8192, "response_mime_type": "text/plain",
            }
            self.model = genai.GenerativeModel(
                model_name=MODEL_ID, generation_config=generation_config,
            )
            print("Google AI Models Initialized.")
        except Exception as e:
            print(f"Error initializing Google AI clients: {e}")
            raise

    def _load_or_build_vector_db(self):
        """Loads the vector database from CSV or builds it if not found."""
        self.vector_db = None
        try:
            if os.path.exists(VECTOR_DB_PATH):
                print(f"Loading existing vector database from {VECTOR_DB_PATH}...")
                self.vector_db = pd.read_csv(VECTOR_DB_PATH)
                if 'embeddings' in self.vector_db.columns and isinstance(self.vector_db['embeddings'].iloc[0], str):
                    print("Converting string embeddings to numpy arrays...")
                    self.vector_db['embeddings'] = self.vector_db['embeddings'].apply(lambda x: np.array(eval(x)))
                elif 'embeddings' in self.vector_db.columns and isinstance(self.vector_db['embeddings'].iloc[0], list):
                     self.vector_db['embeddings'] = self.vector_db['embeddings'].apply(np.array)
                print("Vector database loaded successfully.")
            else:
                print(f"{VECTOR_DB_PATH} not found. Building index...")
                # Ensure DOCUMENTS_PATH_PATTERN is correct relative to execution
                if not glob.glob(DOCUMENTS_PATH_PATTERN):
                     print(f"Warning: No documents found at '{DOCUMENTS_PATH_PATTERN}'. Index cannot be built.")
                     raise FileNotFoundError(f"No documents found matching pattern: {DOCUMENTS_PATH_PATTERN}")

                self.vector_db = self._build_index(
                    DOCUMENTS_PATH_PATTERN,
                    self.embedding_model_name
                )
                vector_db_save = self.vector_db.copy()
                vector_db_save['embeddings'] = vector_db_save['embeddings'].apply(lambda x: x.tolist())
                vector_db_save.to_csv(VECTOR_DB_PATH, index=False)
                print(f"Index built and saved to {VECTOR_DB_PATH}")

        except FileNotFoundError as fnf:
            print(f"Error during Vector DB loading/building: {fnf}")
        except Exception as e:
            print(f"An unexpected error occurred during Vector DB loading/building: {e}")
            import traceback
            traceback.print_exc()

        if self.vector_db is None:
             print("CRITICAL ERROR: Vector database could not be loaded or built.")

    @retry(wait=wait_random_exponential(multiplier=1, max=120), stop=stop_after_attempt(4))
    def _get_embeddings(self, text: str, task_type="RETRIEVAL_DOCUMENT") -> list[float] | None:
        """Generates embeddings for the given text SYNCHRONOUSLY."""
        print(f"Getting embedding for task '{task_type}'...") # Debug print
        try:
            # Use the synchronous version
            response = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type=task_type,
            )
            return response['embedding'] # Return the list directly
        except Exception as e:
            # Check for rate limit errors specifically if possible
            if "resource has been exhausted" in str(e).lower() or "quota" in str(e).lower():
                print(f"Rate limit likely hit during embedding: {e}. Retrying...")
                raise # Re-raise to trigger tenacity retry
            # Handle other potential API errors
            elif "api key not valid" in str(e).lower():
                 print(f"API Key Error during embedding: {e}")
                 # Stop retrying for API key errors
                 return None # Or raise a specific exception
            else:
                print(f"Non-retryable error generating embeddings: {str(e)}")
                return None # Return None for other errors after retries fail


    def _build_index(
        self,
        document_paths_pattern: str,
        embedding_model_name: str,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        separators: list[str] = ["\n\nḤadīth No.", "\n\n", "\n", ". ", ", ", " ", ""],
        sleep_between_embeddings: float = 1.1
    ) -> pd.DataFrame:
        """Builds an index from PDF documents synchronously."""
        all_chunks_data = []
        print(f"Building index: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        print(f"Waiting {sleep_between_embeddings} seconds between embedding calls.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        total_chunks_processed = 0
        total_embeddings_failed = 0
        document_files = glob.glob(document_paths_pattern)

        for doc_path in document_files:
            print(f"Processing document: {doc_path}")
            try:
                with open(doc_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    doc_name = os.path.basename(doc_path)

                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if not page_text: continue
                        except Exception as page_err:
                            print(f"  Error extracting text from page {page_num + 1}: {page_err}")
                            continue

                        chunks = text_splitter.split_text(page_text)
                        for chunk_num, chunk_text in enumerate(chunks):
                            total_chunks_processed += 1
                            if len(chunk_text.strip()) < 20: continue
                            time.sleep(sleep_between_embeddings)
                            try:
                                embeddings = self._get_embeddings(chunk_text, task_type="RETRIEVAL_DOCUMENT")
                            except Exception as emb_err: # Catch retry errors
                                print(f"  Embedding failed permanently for chunk {chunk_num} on page {page_num + 1}: {emb_err}")
                                embeddings = None
                            if embeddings is None:
                                total_embeddings_failed += 1
                                print(f"  Warning: Embedding returned None for chunk {chunk_num} on page {page_num + 1}")
                                continue
                            all_chunks_data.append({
                                "document_name": doc_name, "page_number": page_num + 1,
                                "chunk_number": chunk_num, "chunk_text": chunk_text,
                                "embeddings": embeddings,
                            })
                            if total_chunks_processed % 50 == 0: print(f"  Processed {total_chunks_processed} chunks...")
            except FileNotFoundError: print(f"Error: Document not found at {doc_path}. Skipping."); continue
            except Exception as e: print(f"Error processing document {doc_path}: {str(e)}"); continue

        if not all_chunks_data: raise ValueError("No chunks were successfully created and embedded.")
        print(f"\nFinished processing index build."); print(f"Total chunks attempted: {total_chunks_processed}"); print(f"Total chunks successfully embedded: {len(all_chunks_data)}"); print(f"Total embedding failures (skipped): {total_embeddings_failed}")
        df = pd.DataFrame(all_chunks_data); df['embeddings'] = df['embeddings'].apply(np.array); return df

    def _get_relevant_chunks(self, query: str, top_k: int = 3) -> str:
        """Retrieves relevant text chunks based on query similarity synchronously."""
        if self.vector_db is None: return "Error: Vector database is not available."
        try:
            query_embedding = self._get_embeddings(query, task_type="RETRIEVAL_QUERY")
            if query_embedding is None:
                print("Query embedding failed (returned None).")
                # Provide a more specific error message if possible
                return "Could not process query due to embedding failure (check API key/quota)."

            query_embedding_2d = np.array([query_embedding])
            database_embeddings = np.stack(self.vector_db["embeddings"].to_numpy())

            if query_embedding_2d.shape[1] != database_embeddings.shape[1]:
                raise ValueError(f"Embedding dimension mismatch: Query ({query_embedding_2d.shape[1]}) vs DB ({database_embeddings.shape[1]})")

            similarities = cosine_similarity(query_embedding_2d, database_embeddings)[0]
            # Ensure indices are within bounds
            if len(similarities) < top_k:
                 top_k = len(similarities) # Adjust top_k if fewer chunks exist
                 if top_k == 0: return "No chunks found in the database to compare against." # Handle empty DB

            top_indices = np.argsort(similarities)[-top_k:][::-1]
            relevant_chunks = self.vector_db.iloc[top_indices]
            context = [f"[Page {row['page_number']}, Chunk {row['chunk_number']} from {row['document_name']}]: {row['chunk_text']}" for _, row in relevant_chunks.iterrows()]
            return "\n\n".join(context)
        except ValueError as ve: # Catch specific errors like dimension mismatch
             print(f"ValueError in get_relevant_chunks: {ve}")
             return f"Error retrieving relevant chunks: {ve}"
        except Exception as e:
            print(f"Error getting relevant chunks: {str(e)}")
            import traceback; traceback.print_exc();
            return "Error retrieving relevant chunks."

    def _generate_answer_from_context(self, prompt: str) -> str:
        """Helper to call the generative model SYNCHRONOUSLY."""
        print("Generating content synchronously...") # Debug print
        try:
            # Use the synchronous version
            response = self.model.generate_content(prompt)

            # Refined error/block checking for synchronous response
            if not response.candidates:
                 feedback = getattr(response, 'prompt_feedback', None)
                 block_reason = getattr(feedback, 'block_reason', 'Unknown')
                 print(f"Warning: Response was blocked or had no candidates. Reason: {block_reason}")
                 return f"Could not generate answer. The response was blocked (Reason: {block_reason})."

            # Check if text attribute exists and is not empty
            if not hasattr(response, 'text') or not response.text:
                 print("Warning: Response generated but text is missing or empty.")
                 # You might want to inspect response.candidates[0].content if text is missing
                 try:
                     # Try accessing content parts if text is missing
                     content_parts = response.candidates[0].content.parts
                     if content_parts:
                          return "".join(part.text for part in content_parts if hasattr(part, 'text'))
                     else:
                          return "Could not generate answer. Response content was empty."
                 except (IndexError, AttributeError):
                     return "Could not generate answer. Response structure unexpected."

            return response.text
        except Exception as e:
            print(f"Error during synchronous generation: {e}")
            if "resource_exhausted" in str(e).lower() or "quota" in str(e).lower():
                return "Could not generate answer due to API quota limits."
            elif "api key not valid" in str(e).lower():
                 return "Could not generate answer due to invalid API Key."
            # Handle other potential exceptions from the API client
            return f"An unexpected error occurred during text generation: {str(e)}"

    def _generate_answer(self, query: str, context: str) -> str:
        """Generates an answer using the LLM synchronously."""
        print("Preparing prompt for synchronous generation...") # Debug print
        try:
            if "error" in context.lower() or "could not process" in context.lower():
                 print(f"Skipping generation due to context error: {context}")
                 return "Could not generate answer because context retrieval failed."

            prompt = f"""You are an assistant knowledgeable about Imam Nawawi's 40 Hadith.
                Your task is to answer the user's question based *only* on the provided context below, which is extracted from the book "The 40 Hadith of Al-Imam An-Nawawi".

                Instructions:
                1. Read the context carefully.
                2. If the question asks to explain a specific Hadith (e.g., "explain Hadith No. 1"), synthesize the information from the context that pertains *only* to that specific Hadith number and its explanation/commentary.
                3. Provide a clear and concise explanation based *strictly* on the given text.
                4. Do *not* add any information, commentary, opinions, or interpretations that are not present in the provided context.
                5. If the provided context does *not* contain the information needed to answer the question (e.g., the requested Hadith number is not found in the context), explicitly state that the information is not available in the provided context. Do not attempt to guess or retrieve information from external knowledge.
                6. Structure your answer clearly. If citing page/chunk numbers, use the format [Page X, Chunk Y].

                Context:
                ---
                {context}
                ---

                Question: {query}

                Answer:"""

            response = self._generate_answer_from_context(prompt)
            return response

        except Exception as e:
            print(f"Error in _generate_answer wrapper: {str(e)}")
            return f"An unexpected error occurred while preparing the answer generation: {str(e)}"

    def _transform_query(self, query: str) -> str:
        """Transforms natural language numbers in the query."""
        query_lower = query.lower()
        # (Keep the extensive replacements dictionary)
        replacements = {
            "first hadith": "Hadith No. 1", "hadith number one": "Hadith No. 1", "hadith number 1": "Hadith No. 1", "ḥadīth no. 1": "Hadith No. 1",
            "second hadith": "Hadith No. 2", "hadith number two": "Hadith No. 2", "hadith number 2": "Hadith No. 2", "ḥadīth no. 2": "Hadith No. 2",
            "third hadith": "Hadith No. 3", "hadith number three": "Hadith No. 3", "hadith number 3": "Hadith No. 3", "ḥadīth no. 3": "Hadith No. 3",
            "fourth hadith": "Hadith No. 4", "hadith number four": "Hadith No. 4", "hadith number 4": "Hadith No. 4", "ḥadīth no. 4": "Hadith No. 4",
            "fifth hadith": "Hadith No. 5", "hadith number five": "Hadith No. 5", "hadith number 5": "Hadith No. 5", "ḥadīth no. 5": "Hadith No. 5",
            "sixth hadith": "Hadith No. 6", "hadith number six": "Hadith No. 6", "hadith number 6": "Hadith No. 6", "ḥadīth no. 6": "Hadith No. 6",
            "seventh hadith": "Hadith No. 7", "hadith number seven": "Hadith No. 7", "hadith number 7": "Hadith No. 7", "ḥadīth no. 7": "Hadith No. 7",
            "eighth hadith": "Hadith No. 8", "hadith number eight": "Hadith No. 8", "hadith number 8": "Hadith No. 8", "ḥadīth no. 8": "Hadith No. 8",
            "ninth hadith": "Hadith No. 9", "hadith number nine": "Hadith No. 9", "hadith number 9": "Hadith No. 9", "ḥadīth no. 9": "Hadith No. 9",
            "tenth hadith": "Hadith No. 10", "hadith number ten": "Hadith No. 10", "hadith number 10": "Hadith No. 10", "ḥadīth no. 10": "Hadith No. 10",
            "eleventh hadith": "Hadith No. 11", "hadith number eleven": "Hadith No. 11", "hadith number 11": "Hadith No. 11", "ḥadīth no. 11": "Hadith No. 11",
            "twelfth hadith": "Hadith No. 12", "hadith number twelve": "Hadith No. 12", "hadith number 12": "Hadith No. 12", "ḥadīth no. 12": "Hadith No. 12",
            "thirteenth hadith": "Hadith No. 13", "hadith number thirteen": "Hadith No. 13", "hadith number 13": "Hadith No. 13", "ḥadīth no. 13": "Hadith No. 13",
            "fourteenth hadith": "Hadith No. 14", "hadith number fourteen": "Hadith No. 14", "hadith number 14": "Hadith No. 14", "ḥadīth no. 14": "Hadith No. 14",
            "fifteenth hadith": "Hadith No. 15", "hadith number fifteen": "Hadith No. 15", "hadith number 15": "Hadith No. 15", "ḥadīth no. 15": "Hadith No. 15",
            "sixteenth hadith": "Hadith No. 16", "hadith number sixteen": "Hadith No. 16", "hadith number 16": "Hadith No. 16", "ḥadīth no. 16": "Hadith No. 16",
            "seventeenth hadith": "Hadith No. 17", "hadith number seventeen": "Hadith No. 17", "hadith number 17": "Hadith No. 17", "ḥadīth no. 17": "Hadith No. 17",
            "eighteenth hadith": "Hadith No. 18", "hadith number eighteen": "Hadith No. 18", "hadith number 18": "Hadith No. 18", "ḥadīth no. 18": "Hadith No. 18",
            "nineteenth hadith": "Hadith No. 19", "hadith number nineteen": "Hadith No. 19", "hadith number 19": "Hadith No. 19", "ḥadīth no. 19": "Hadith No. 19",
            "twentieth hadith": "Hadith No. 20", "hadith number twenty": "Hadith No. 20", "hadith number 20": "Hadith No. 20", "ḥadīth no. 20": "Hadith No. 20",
            "twenty-first hadith": "Hadith No. 21", "hadith number twenty-one": "Hadith No. 21", "hadith number 21": "Hadith No. 21", "ḥadīth no. 21": "Hadith No. 21",
            "twenty-second hadith": "Hadith No. 22", "hadith number twenty-two": "Hadith No. 22", "hadith number 22": "Hadith No. 22", "ḥadīth no. 22": "Hadith No. 22",
            "twenty-third hadith": "Hadith No. 23", "hadith number twenty-three": "Hadith No. 23", "hadith number 23": "Hadith No. 23", "ḥadīth no. 23": "Hadith No. 23",
            "twenty-fourth hadith": "Hadith No. 24", "hadith number twenty-four": "Hadith No. 24", "hadith number 24": "Hadith No. 24", "ḥadīth no. 24": "Hadith No. 24",
            "twenty-fifth hadith": "Hadith No. 25", "hadith number twenty-five": "Hadith No. 25", "hadith number 25": "Hadith No. 25", "ḥadīth no. 25": "Hadith No. 25",
            "twenty-sixth hadith": "Hadith No. 26", "hadith number twenty-six": "Hadith No. 26", "hadith number 26": "Hadith No. 26", "ḥadīth no. 26": "Hadith No. 26",
            "twenty-seventh hadith": "Hadith No. 27", "hadith number twenty-seven": "Hadith No. 27", "hadith number 27": "Hadith No. 27", "ḥadīth no. 27": "Hadith No. 27",
            "twenty-eighth hadith": "Hadith No. 28", "hadith number twenty-eight": "Hadith No. 28", "hadith number 28": "Hadith No. 28", "ḥadīth no. 28": "Hadith No. 28",
            "twenty-ninth hadith": "Hadith No. 29", "hadith number twenty-nine": "Hadith No. 29", "hadith number 29": "Hadith No. 29", "ḥadīth no. 29": "Hadith No. 29",
            "thirtieth hadith": "Hadith No. 30", "hadith number thirty": "Hadith No. 30", "hadith number 30": "Hadith No. 30", "ḥadīth no. 30": "Hadith No. 30",
            "thirty-first hadith": "Hadith No. 31", "hadith number thirty-one": "Hadith No. 31", "hadith number 31": "Hadith No. 31", "ḥadīth no. 31": "Hadith No. 31",
            "thirty-second hadith": "Hadith No. 32", "hadith number thirty-two": "Hadith No. 32", "hadith number 32": "Hadith No. 32", "ḥadīth no. 32": "Hadith No. 32",
            "thirty-third hadith": "Hadith No. 33", "hadith number thirty-three": "Hadith No. 33", "hadith number 33": "Hadith No. 33", "ḥadīth no. 33": "Hadith No. 33",
            "thirty-fourth hadith": "Hadith No. 34", "hadith number thirty-four": "Hadith No. 34", "hadith number 34": "Hadith No. 34", "ḥadīth no. 34": "Hadith No. 34",
            "thirty-fifth hadith": "Hadith No. 35", "hadith number thirty-five": "Hadith No. 35", "hadith number 35": "Hadith No. 35", "ḥadīth no. 35": "Hadith No. 35",
            "thirty-sixth hadith": "Hadith No. 36", "hadith number thirty-six": "Hadith No. 36", "hadith number 36": "Hadith No. 36", "ḥadīth no. 36": "Hadith No. 36",
            "thirty-seventh hadith": "Hadith No. 37", "hadith number thirty-seven": "Hadith No. 37", "hadith number 37": "Hadith No. 37", "ḥadīth no. 37": "Hadith No. 37",
            "thirty-eighth hadith": "Hadith No. 38", "hadith number thirty-eight": "Hadith No. 38", "hadith number 38": "Hadith No. 38", "ḥadīth no. 38": "Hadith No. 38",
            "thirty-ninth hadith": "Hadith No. 39", "hadith number thirty-nine": "Hadith No. 39", "hadith number 39": "Hadith No. 39", "ḥadīth no. 39": "Hadith No. 39",
            "fortieth hadith": "Hadith No. 40", "hadith number forty": "Hadith No. 40", "hadith number 40": "Hadith No. 40", "ḥadīth no. 40": "Hadith No. 40",
            "forty-first hadith": "Hadith No. 41", "hadith number forty-one": "Hadith No. 41", "hadith number 41": "Hadith No. 41", "ḥadīth no. 41": "Hadith No. 41",
            "forty-second hadith": "Hadith No. 42", "hadith number forty-two": "Hadith No. 42", "hadith number 42": "Hadith No. 42", "ḥadīth no. 42": "Hadith No. 42",
        }
        transformed_query = query
        for key, value in replacements.items():
            if key in query_lower:
                pattern = re.compile(re.escape(key), re.IGNORECASE)
                transformed_query = pattern.sub(value, transformed_query, count=1)
                break
        return transformed_query

    def process_query(self, question: str, top_k: int = 3) -> Tuple[str, str]:
        """
        Processes a user query using the RAG pipeline SYNCHRONOUSLY.
        """
        if self.vector_db is None:
            return "Error: Vector database is not initialized.", "Initialization Error"

        start_time = time.time()
        answer = "Processing failed."
        source_info = "N/A"
        relevant_context = "Context retrieval pending."

        try:
            transformed_question = self._transform_query(question)
            print(f"Original query: '{question}' -> Transformed query: '{transformed_question}'")

            relevant_context = self._get_relevant_chunks(transformed_question, top_k=top_k)
            print(f"Retrieved context snippet: '{str(relevant_context)[:200]}...'")

            if not relevant_context or "error" in relevant_context.lower() or "could not process" in relevant_context.lower():
                print(f"Context retrieval failed or returned error: {relevant_context}")
                answer = "Could not generate answer because context retrieval failed."
                source_info = f"Context Retrieval Failed: {relevant_context}" # Include reason if possible
            else:
                print("Generating answer synchronously...")
                # Call the synchronous version
                answer = self._generate_answer(question, relevant_context)
                try:
                    first_line = relevant_context.split('\n\n')[0]
                    if ':' in first_line and '[' in first_line and ']' in first_line: source_info = first_line.split(':', 1)[0].strip('[] ')
                    else: source_info = "Retrieved Context (Format Unknown)"
                except Exception: source_info = "Source Parsing Error"

            end_time = time.time(); print(f"Processing time: {end_time - start_time:.2f} seconds"); print(f"Generated answer snippet: {answer[:200]}...")
            return answer, source_info
        except Exception as e:
            print(f"Error processing query '{question}': {str(e)}"); import traceback; traceback.print_exc();
            error_context_info = f"(Context snippet before error: '{str(relevant_context)[:100]}...')"; answer = f"Error processing question: {str(e)} {error_context_info}"
            return answer, "Processing Error"

# --- End of HadidRAG Class ---