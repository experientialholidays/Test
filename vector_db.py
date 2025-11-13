import os
import shutil
import pandas as pd
import ast  
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv(override=True)


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

    def load_documents(self):
        documents = []
        for file in os.listdir(self.folder):
            file_path = os.path.join(self.folder, file)
            source_file = os.path.basename(file)

            if file_path.endswith(".xlsx") and not file.startswith("~$"):
                xls = pd.ExcelFile(file_path)
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    df.columns = df.columns.str.lower() # Normalize headers to lowercase

                    for index, row in df.iterrows():
                        row_text = ", ".join([str(x) for x in row.tolist()])

                        # --- MAPPING NEW EXCEL HEADERS TO AGENT METADATA KEYS ---
                        # REQUIRED FOR FILTERING/SUMMARY
                        day_raw = str(row.get("days", "N/A"))
                        date = str(row.get("dates", "N/A"))
                        location = str(row.get("venue", "N/A"))
                        
                        # REQUIRED FOR FULL CARD DISPLAY
                        title = str(row.get("event name", "N/A"))
                        time = str(row.get("times", "N/A"))
                        contribution = str(row.get("cost/contribution", "N/A"))
                        contact_info = str(row.get("contact person/unit", "")).strip()
                        phone_number = str(row.get("contact phone/whatsapp", "")).strip() 
                        poster_url = str(row.get("website/link", None))
                        # --- END MAPPING ---

                        # NEW LOGIC: expand multi-day cells like '["Monday", "Tuesday"]'
                        if isinstance(day_raw, str) and day_raw.startswith("[") and day_raw.endswith("]"):
                            try:
                                day_list = ast.literal_eval(day_raw)
                                if not isinstance(day_list, list):
                                    day_list = [day_raw]
                            except Exception:
                                day_list = [day_raw]
                        else:
                            # allow comma-separated or single day
                            day_list = [d.strip() for d in day_raw.split(",") if d.strip()] or ["N/A"]

                        # create one Document per day
                        for single_day in day_list:
                            final_poster_url = poster_url if poster_url != 'None' else None
                            
                            documents.append(
                                Document(
                                    page_content=row_text,
                                    metadata={
                                        "source": source_file,
                                        "sheet": sheet_name,
                                        "day": single_day.strip(),
                                        "date": date.strip(),
                                        "location": location.strip(),
                                        # --- AGENT METADATA KEYS ---
                                        "title": title.strip(),
                                        "time": time.strip(),
                                        "contribution": contribution.strip(),
                                        "contact": contact_info,
                                        "poster_url": final_poster_url, 
                                        "phone": phone_number,
                                        # --- END AGENT METADATA KEYS ---
                                    },
                                )
                            )

            elif file_path.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = source_file
                    doc.metadata["page"] = doc.metadata.get("page", "N/A")
                    # Add required defaults for PDF/Text files
                    doc.metadata.update({
                        "day": "N/A", "date": "N/A", "location": "N/A",
                        "title": "N/A", "time": "N/A", "contribution": "N/A",
                        "contact": "", "poster_url": None, "phone": ""
                    })
                    documents.append(doc)

            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = source_file
                    # Add required defaults for PDF/Text files
                    doc.metadata.update({
                        "day": "N/A", "date": "N/A", "location": "N/A",
                        "title": "N/A", "time": "N/A", "contribution": "N/A",
                        "contact": "", "poster_url": None, "phone": ""
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
