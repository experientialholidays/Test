import os
from dotenv import load_dotenv
import gradio as gr

# LangChain + Chroma + OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, UnstructuredFileLoader, UnstructuredExcelLoader, TextLoader
from langchain.prompts import PromptTemplate

# -----------------------------
# CONFIG
# -----------------------------
MODEL = "gpt-5"
DB_NAME = "vector_db"

# Load environment variables
load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-fallback-key")

# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
#PDF_PATH = "https://huggingface.co/spaces/varunnick/app.py/resolve/main/1096%2011%20Sept%202025.pdf"  # change if deploying
#loader = PyMuPDFLoader(PDF_PATH)
#documents = loader.load()

folder = "Tarun"


# Custom mapping: extension -> loader class
def custom_loader(file_path: str):
    if file_path.endswith(".pdf"):
        return PyMuPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        return TextLoader(file_path)
    elif file_path.endswith(".xlsx"):
        return UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

loader = DirectoryLoader(
    folder,
    loader_cls=custom_loader,
    recursive=False  # set True if you want to scan subfolders
)

# -----------------------------
# EMBEDDINGS + VECTORSTORE
# -----------------------------
embeddings = OpenAIEmbeddings()

# Check if vectorstore exists, if not create it
if os.path.exists(DB_NAME):
    print("Loading existing vector database...")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
else:
    print("Creating new vector database...")
    # Load documents and create chunks only when creating new database
    documents = loader.load()
    
    # -----------------------------
    # SPLIT INTO CHUNKS
    # -----------------------------
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Create new vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME)


# -----------------------------
# CHAT SETUP
# -----------------------------
# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=1, model_name=MODEL)

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG; k is how many chunks to use
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# -----------------------------
# GRADIO APP
# -----------------------------
def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]

if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat,
        type="messages",
        title="Auroville Events Chatbot",
        description="Ask me anything about events and activities in Auroville during September",
        examples=[["Hi! 👋 I'm your AI assistant. Ask me anything events and activities in Auroville during September"]],
    ).launch()