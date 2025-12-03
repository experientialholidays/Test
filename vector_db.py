import os
import shutil
import pandas as pd
import ast
import re
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
            "poster url": "poster_url",
            "Category": "category", 
            # NEW HEADERS ADDED HERE
            "Description": "description",
            "Contact Email": "email",
            "Target Audience/Prerequisites": "audience",
        }

    def load_documents(self):
        # --- SAFETY HELPER ---
        def cell_to_str(val):
            """Convert Excel cell values to clean strings."""
            if val is None: return ""
            try:
                if pd.isna(val): return ""
            except Exception: pass
            if isinstance(val, float) and str(val) == "nan": return ""
            if isinstance(val, datetime): return val.strftime("%B %d, %Y")
            if isinstance(val, str): return val.strip()
            if isinstance(val, (int, float)): return str(val)
            try: return str(val).strip()
            except Exception: return ""

        # --- DATE PARSING HELPER (kept for context, no change) ---
        def parse_date_to_iso_range(date_str):
            """
            Parses a date string (single or range) into ISO start/end dates.
            Returns tuple: (start_date_iso, end_date_iso) or ("", "")
            """
            if not date_str:
                return "", ""

            s = str(date_str).strip()
            year = datetime.now().year
            formats = ["%B %d, %Y", "%B %d", "%d %B", "%d %b", "%d.%m.%y", "%d.%m.%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"]

            def try_parse(ds):
                for fmt in formats:
                    try:
                        p = ds
                        if "%Y" not in fmt and "%y" not in fmt:
                            p = f"{ds}, {year}"
                        return datetime.strptime(p.strip(), fmt.strip()).date()
                    except:
                        continue
                return None

            # 1. Try Regex for "17-28 November" pattern
            range_match = re.search(r'^(\d{1,2})\s*[—–-]\s*(\d{1,2})\s+([A-Za-z]+)$', s)
            if range_match:
                start_day, end_day, month_str = range_match.groups()
                start_full = f"{start_day} {month_str}"
                end_full = f"{end_day} {month_str}"
                
                sd = try_parse(start_full)
                ed = try_parse(end_full)
                if sd and ed:
                    return sd.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d")

            # 2. Standard Split (e.g., "Nov 17 - Nov 28")
            parts = re.split(r'\s*[—–-]\s*|\s+to\s+', s)
            
            if len(parts) == 2:
                start_obj = try_parse(parts[0])
                end_obj = try_parse(parts[1])
                if start_obj and end_obj:
                    return start_obj.strftime("%Y-%m-%d"), end_obj.strftime("%Y-%m-%d")
            
            # 3. Single Date
            elif len(parts) == 1:
                single_obj = try_parse(parts[0])
                if single_obj:
                    iso = single_obj.strftime("%Y-%m-%d")
                    return iso, iso

            return "", ""

        # ------------------- LOADING LOGIC -------------------

        documents = []
        for file in os.listdir(self.folder):
            file_path = os.path.join(self.folder, file)
            source_file = os.path.basename(file)

            if file_path.endswith(".xlsx") and not file.startswith("~$"):
                xls = pd.ExcelFile(file_path)
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    df.columns = df.columns.str.lower()
                    df = df.fillna('')

                    # Convert all relevant columns to string for consistent handling
                    for col_key in self.EXCEL_HEADERS.keys():
                         # Need to check for the lowercase version of the header name as column names were lowered earlier
                        col_lower = col_key.lower()
                        if col_lower in df.columns:
                            # Use original header key for lookup in row.get() later
                            df[col_lower] = df[col_lower].astype(str)

                    for index, row in df.iterrows():
                        # Use lowercase keys for robust row lookup
                        row_text = ", ".join([str(x) for x in row.tolist()])

                        # Existing Clean values
                        day_raw = cell_to_str(row.get("days", ""))
                        date = cell_to_str(row.get("dates", ""))
                        location = cell_to_str(row.get("venue", ""))
                        title = cell_to_str(row.get("event name", ""))
                        time_str = cell_to_str(row.get("times", ""))
                        contribution = cell_to_str(row.get("cost/contribution", ""))
                        contact_info = cell_to_str(row.get("contact person/unit", ""))
                        phone_number = cell_to_str(row.get("contact phone/whatsapp", ""))
                        category = cell_to_str(row.get("category", ""))
                        poster_url_raw = cell_to_str(row.get("poster url", "")) # Corrected to "poster url" per EXCEL_HEADERS
                        poster_url = poster_url_raw if poster_url_raw else None

                        # --- NEW: Extract and Clean New Values ---
                        description = cell_to_str(row.get("description", ""))
                        contact_email = cell_to_str(row.get("contact email", ""))
                        target_audience = cell_to_str(row.get("target audience/prerequisites", ""))
                        # ----------------------------------------

                        # Generate ISO Start/End Meta
                        start_date_meta, end_date_meta = parse_date_to_iso_range(date)

                        # Multi-day handling
                        if day_raw.startswith("[") and day_raw.endswith("]"):
                            try:
                                day_list = ast.literal_eval(day_raw)
                                if not isinstance(day_list, list):
                                    day_list = [day_raw]
                            except Exception:
                                day_list = [day_raw]
                        else:
                            day_list = [d.strip() for d in day_raw.split(",") if d.strip()] or [""]

                        for single_day in day_list:
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
                                        "day": final_day,
                                        "date": final_date,
                                        "location": final_location,
                                        "title": final_title,
                                        "time": time_str if time_str else "",
                                        "contribution": contribution if contribution else "",
                                        "contact": contact_info,
                                        "poster_url": poster_url,
                                        "phone": phone_number,
                                        "category": category if category else "",
                                        # DATE METADATA
                                        "start_date_meta": start_date_meta,
                                        "end_date_meta": end_date_meta,
                                        # NEW METADATA FIELDS ADDED HERE
                                        "description": description,
                                        "email": contact_email,
                                        "audience": target_audience,
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
                        "category": "",
                        "start_date_meta": "", "end_date_meta": "",
                        # NEW METADATA DEFAULTS
                        "description": "", "email": "", "audience": ""
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
                        "category": "",
                        "start_date_meta": "", "end_date_meta": "",
                        # NEW METADATA DEFAULTS
                        "description": "", "email": "", "audience": ""
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
