import os
import shutil
import pandas as pd
import ast  
from datetime import datetime, time
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv(override=True)

# -------------------------------------------------------------------------

class VectorDBManager:
    def __init__(self, folder="input", db_name="vector_db", chunk_size=2000, chunk_overlap=200):
        self.folder = folder
        self.db_name = db_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.vectorstore = None
        
        # --- Define the required Excel headers (in lowercase for robust lookup) ---
        self.EXCEL_HEADERS = {
            "Event Name": "title", 
            "Days": "day", 
            "Dates": "date", 
            "Times": "time", 
            "Venue": "location", 
            "Cost/Contribution": "contribution", 
            "Contact Person/Unit": "contact", 
            "Contact Phone/Whatsapp": "phone", 
            "Website/Link": "poster_url",
            "Category": "category", # <-- ADDED
        }

    def load_documents(self):
        documents = []
        for file in os.listdir(self.folder):
            file_path = os.path.join(self.folder, file)
            source_file = os.path.basename(file)

            if file_path.endswith(".xlsx") and not file.startswith("~$"):
                xls = pd.ExcelFile(file_path)
                for sheet_name in xls.sheet_names:
                    # Read the sheet and convert headers to lowercase
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    df.columns = df.columns.str.lower()
                    
                    # --- CRITICAL FIX: Replace all NaN values with EMPTY STRING ("") ---
                    df = df.fillna('')

                    # Force critical columns to string type
                    for col in self.EXCEL_HEADERS.keys():
                         if col in df.columns:
                            df[col] = df[col].astype(str)
                    
                    for index, row in df.iterrows():
                        # Create full text content for embedding
                        row_text = ", ".join([str(x) for x in row.tolist()])

                        # --- MAPPING: Extract and strip critical fields, defaulting to "" ---
                        day_raw = row.get("days", "").strip()
                        date = row.get("dates", "").strip()
                        location = row.get("venue", "").strip()
                        
                        title = row.get("event name", "").strip()
                        time_str = row.get("times", "").strip()
                        contribution = row.get("cost/contribution", "").strip()
                        
                        contact_info = row.get("contact person/unit", "").strip()
                        phone_number = row.get("contact phone/whatsapp", "").strip() 
                        category = row.get("category", "").strip() # <-- ADDED
                        
                        # Handle poster_url separately: None is better than "" if it's blank
                        poster_url_raw = row.get("website/link", "").strip()
                        poster_url = poster_url_raw if poster_url_raw else None
                        
                        # Logic to expand multi-day cells (e.g., '["Monday", "Tuesday"]')
                        if day_raw.startswith("[") and day_raw.endswith("]"):
                            try:
                                day_list = ast.literal_eval(day_raw)
                                if not isinstance(day_list, list):
                                    day_list = [day_raw]
                            except Exception:
                                day_list = [day_raw]
                        else:
                            day_list = [d.strip() for d in day_raw.split(",") if d.strip()] or [""]

                        # Create one Document per day
                        for single_day in day_list:
                            
                            # Final metadata cleanup: ensure empty strings are used for missing values
                            final_day = single_day if single_day else ""
                            final_date = date if date else ""
                            final_location = location if location else ""
                            final_title = title if title else "Event"

                            documents.append(
                                Document(
                                    page_content=row_text,
                                    metadata={
                                        "source": source_file,
                                        "sheet": sheet_name,
                                        # --- AGENT METADATA KEYS ---
                                        "day": final_day,
                                        "date": final_date,
                                        "location": final_location,
                                        "title": final_title,
                                        "time": time_str if time_str else "",
                                        "contribution": contribution if contribution else "",
                                        "contact": contact_info,
                                        "poster_url": poster_url, 
                                        "phone": phone_number,
                                        "category": category if category else "", # <-- ADDED
                                        # --- END AGENT METADATA KEYS ---
                                    },
                                )
                            )

            elif file_path.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = source_file
                    doc.metadata["page"] = doc.metadata.get("page", "")
                    doc.metadata.update({
                        "day": "", "date": "", "location": "",
                        "title": "Document Content", "time": "", "contribution": "",
                        "contact": "", "poster_url": None, "phone": "",
                        "category": "" # <-- ADDED
                    })
                    documents.append(doc)

            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = source_file
                    doc.metadata.update({
                        "day": "", "date": "", "location": "",
                        "title": "Document Content", "time": "", "contribution": "",
                        "contact": "", "poster_url": None, "phone": "",
                        "category": "" # <-- ADDED
                    })
                    documents.append(doc)

        return documents

    def create_or_load_db(self, force_refresh=False):
        if os.path.exists(self.db_name) and not force_refresh:
            print("Loading existing vector database with metadata support (Fast Path)...")
            self.vectorstore = Chroma(
                persist_directory=self.db_name,
                embedding_function=self.embeddings,
            )
        else:
            if force_refresh:
                print("Force refresh enabled - recreating vector database (Slow Path)...")
                shutil.rmtree(self.db_name, ignore_errors=True)
            else:
                 print("Creating new vector database (Slow Path)...")

            documents = self.load_documents()

            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Loaded {len(documents)} documents and split into {len(chunks)} chunks")

            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.db_name,
            )

        return self.vectorstore

    def get_retriever(self, k=50):
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized. Call create_or_load_db() first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
