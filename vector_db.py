import os
import shutil
import pandas as pd
import ast  # ✅ added to safely parse list-style strings like '["Monday", "Tuesday"]'
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
                    df.columns = df.columns.str.lower()

                    for index, row in df.iterrows():
                        row_text = ", ".join([str(x) for x in row.tolist()])

                        # extract raw metadata
                        day_raw = str(row.get("day", "N/A"))
                        date = str(row.get("date", "N/A"))
                        location = str(row.get("location", "N/A"))

                        # ✅ NEW LOGIC: expand multi-day cells like '["Monday", "Tuesday"]'
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

                        # ✅ create one Document per day
                        for single_day in day_list:
                            documents.append(
                                Document(
                                    page_content=row_text,
                                    metadata={
                                        "source": source_file,
                                        "sheet": sheet_name,
                                        "day": single_day,
                                        "date": date.strip(),
                                        "location": location.strip(),
                                    },
                                )
                            )

            elif file_path.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = source_file
                    doc.metadata["page"] = doc.metadata.get("page", "N/A")
                    doc.metadata["day"] = "N/A"
                    doc.metadata["date"] = "N/A"
                    doc.metadata["location"] = "N/A"
                    documents.append(doc)

            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = source_file
                    doc.metadata["day"] = "N/A"
                    doc.metadata["date"] = "N/A"
                    doc.metadata["location"] = "N/A"
                    documents.append(doc)

        return documents

    def create_or_load_db(self, force_refresh=False):
        if os.path.exists(self.db_name) and not force_refresh:
            print("Loading existing vector database with metadata support...")
            self.vectorstore = Chroma(
                persist_directory=self.db_name,
                embedding_function=self.embeddings,
            )
        else:
            if force_refresh:
                print("Force refresh enabled - recreating vector database...")
                shutil.rmtree(self.db_name, ignore_errors=True)

            print("Creating new vector database with metadata...")
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
