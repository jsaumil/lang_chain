from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize embeddings
embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# Get the absolute path to your document
current_dir = os.path.dirname(os.path.abspath(__file__))
doc_path = os.path.join(current_dir, "docs.txt")

# Check if file exists
if not os.path.exists(doc_path):
    raise FileNotFoundError(f"The file {doc_path} does not exist. Please ensure the file exists and the path is correct.")

# Load the text file
loader = TextLoader(doc_path)
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings and store in FAISS
vectorstore = FAISS.from_documents(docs, embedding)

# Create a retriever (fetches relevant documents)
retriever = vectorstore.as_retriever()

# Manually Retrieve Relevant Documents
query = "What are the key takeaways from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

# Combine Retrieved text into a single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Manually Pass Retrieved Text to LLM
prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"
answer = llm.invoke(prompt)

# Print the answer
print("Answer:", answer)