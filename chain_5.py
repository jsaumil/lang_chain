import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any

# LangChain and related imports
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Other libraries for Memory Manager
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import langchain
from langchain.cache import InMemoryCache

# Set up LangChain cache
langchain.llm_cache = InMemoryCache()

# --- Environment Setup ---
# Make sure to set your GOOGLE_API_KEY as an environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or 'AIzaSyBZbwKnkhqOIXFJT2AwVWC2FbaBf_4sHb4'
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


summary_of_pdf = """A comprehensive guide to the admission process at Gandhinagar University outlines a variety of programs, each with specific eligibility criteria, fee structures, and admission procedures. The university offers a broad spectrum of undergraduate, postgraduate, diploma, and certificate courses across several disciplines.

### **Engineering Programs**

The university offers a 4-year Bachelor of Technology (B.Tech) program with a tuition fee of Rs. 31,000 per semester. Admission is handled by the Admission Committee for Professional Courses (ACPC), and applicants must have completed their 10+2 with Physics, Chemistry, and Mathematics (PCM) and passed the GUJCET. Specializations include Computer Engineering (300 seats), Information Technology (180 seats), Mechanical Engineering (180 seats), Civil Engineering (90 seats), and several others with varying seat capacities.

### **Business Administration Programs**

The Bachelor of Business Administration (BBA) is a 3-year program requiring a 10+2 qualification with English and Accountancy. The 2-year Master of Business Administration (MBA) program is open to graduates who have taken entrance exams like CMAT, MAT, XAT, or CAT.

### **Important Admission Information**

Admissions are conducted either by the ACPC for professional courses or by the Gandhinagar University Admission Committee for most other programs, primarily based on merit. All stated fees are per semester unless specified otherwise and are subject to change. Applicants must have completed their previous education from a recognized board or university."""


# ==============================================================================
# 1. MEMORY MANAGER CLASS
# ==============================================================================

class MemoryManager:
    """Advanced memory management system with short-term, long-term, and episodic memory."""

    def __init__(self, persistence_file: str = "memory.json"):
        self.short_term_memory = ConversationBufferMemory(k=5)
        self.long_term_memory = []
        self.episodic_memory = {}  # Key: session_id, Value: list of conversations
        self.current_session = str(uuid.uuid4())
        self.persistence_file = persistence_file
        self.episodic_memory[self.current_session] = []
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_memory()

    def add_conversation(self, user_input: str, ai_response: str):
        """Add a conversation to all memory systems."""
        timestamp = datetime.now().isoformat()
        conversation = {
            "user_input": user_input,
            "ai_response": ai_response,
            "timestamp": timestamp
        }
        self.short_term_memory.save_context({"input": user_input}, {"output": ai_response})
        if not self._is_duplicated(conversation):
            self.long_term_memory.append(conversation)
            self.episodic_memory[self.current_session].append(conversation)
            if len(self.long_term_memory) > 50:
                self.long_term_memory.pop(0)
        self.save_memory()

    def _is_duplicated(self, new_conversation: Dict) -> bool:
        """Check for similarity to avoid storing redundant information."""
        if not self.long_term_memory:
            return False
        recent_conversations = self.long_term_memory[-5:]
        new_text = f"{new_conversation['user_input']} {new_conversation['ai_response']}"
        new_embedding = self.embedding_model.encode(new_text)
        for conv in recent_conversations:
            existing_text = f"{conv['user_input']} {conv['ai_response']}"
            existing_embedding = self.embedding_model.encode(existing_text)
            if cosine_similarity([new_embedding], [existing_embedding])[0][0] > 0.95:
                return True
        return False

    def get_context(self, query: str, num_conversations: int = 3) -> str:
        """Retrieve relevant conversation history for context."""
        if not self.long_term_memory:
            return "No recent conversation history."
        recent = self.episodic_memory.get(self.current_session, [])[-num_conversations:]
        context_str = "This is the conversation history:\n"
        for conv in recent:
            context_str += f"User: {conv['user_input']}\nAI: {conv['ai_response']}\n"
        return context_str

    def save_memory(self):
        """Save memory to a JSON file."""
        memory_data = {
            "long_term_memory": self.long_term_memory,
            "episodic_memory": self.episodic_memory,
            "current_session": self.current_session
        }
        with open(self.persistence_file, 'w') as f:
            json.dump(memory_data, f, indent=2)

    def load_memory(self):
        """Load memory from a JSON file."""
        if os.path.exists(self.persistence_file):
            with open(self.persistence_file, 'r') as f:
                memory_data = json.load(f)
            self.long_term_memory = memory_data.get("long_term_memory", [])
            self.episodic_memory = memory_data.get("episodic_memory", {})
            self.current_session = memory_data.get("current_session", str(uuid.uuid4()))
            if self.current_session not in self.episodic_memory:
                self.episodic_memory[self.current_session] = []

    def start_new_session(self):
        """Start a new conversation session."""
        self.current_session = str(uuid.uuid4())
        self.episodic_memory[self.current_session] = []
        self.short_term_memory.clear()
        self.save_memory()
        print("--- New Session Started ---")

    def get_memory_stats(self) -> Dict:
        """Get statistics about memory usage."""
        return {
            "long_term_memory_count": len(self.long_term_memory),
            "episodic_memory_sessions": len(self.episodic_memory),
            "current_session_conversations": len(self.episodic_memory.get(self.current_session, [])),
        }

# ==============================================================================
# 2. CONVERSATIONAL RAG CHAIN CLASS (MODIFIED)
# ==============================================================================

class ConversationalRAGChain:
    def __init__(self, pdf_path: str):
        self.memory_manager = MemoryManager()
        self.model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.7)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.parser = StrOutputParser()

        print("Loading and processing PDF...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200))
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        print("PDF processed successfully.")

        # --- Define Component Chains ---

        # 1. Classifier Chain to determine user intent
        classifier_prompt = PromptTemplate(
            template='Is the following question about Gandhinagar University admissions? Answer only with the word "admission" or "general":\n{input}',
            input_variables=['input'],
        )
        self.classifier_chain = classifier_prompt | self.model | self.parser

        # 2. RAG Chain for admission-specific questions (uses PDF context)
        rag_prompt = PromptTemplate(
            template="""You are an expert Gandhinagar University admissions counselor.
Answer the user's question based *only* on the provided context from documents and the conversation history.
If the answer is not in the context, say so clearly.

CONVERSATION HISTORY:
{chat_history}

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {input}

ANSWER:""",
            input_variables=['chat_history', 'context', 'input'],
        )
        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_docs(self.retriever.invoke(x["input"]))
            ) | rag_prompt | self.model | self.parser
        )

        # 3. General Chain for other questions (uses summary context)
        general_prompt = PromptTemplate(
            template="""You are an expert and helpful admissions counselor for Gandhinagar University. Your primary goal is to assist prospective students. Follow these instructions carefully:
Prioritize the Summary: First, always try to answer the user's question using the facts from the SUMMARY CONTEXT provided below. This is your main source of information.
Handle Missing Information Gracefully: If the SUMMARY CONTEXT does not contain the answer, do not say "I don't know" or "The information is not available." Instead, provide a general, helpful answer that would be typical for a university. You must frame this general information as if it applies to Gandhinagar University.
Use Conversation History: Refer to the CONVERSATION HISTORY to understand the context of the user's questions and provide relevant, follow-up answers.

SUMMARY CONTEXT:
{summary_context}

CONVERSATION HISTORY:
{chat_history}

QUESTION: {input}

ANSWER:""",
            input_variables=['chat_history', 'input','summary_context'],
        )
        general_chain = general_prompt | self.model | self.parser

        # 4. Branch to route based on the classifier's output
        # This logic ensures that if a question is about admissions, it ONLY uses the rag_chain.
        self.branch = RunnableBranch(
            (lambda x: "admission" in x["topic"].lower(), rag_chain),
            general_chain,
        )

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    def invoke(self, user_input: str) -> str:
        """
        Orchestrates the response by first classifying the input, printing the
        chosen path, and then invoking the correct chain.
        """
        # Get the conversation history from memory
        chat_history = self.memory_manager.get_context(user_input)

        # 1. First, classify the user's input to determine the topic
        topic = self.classifier_chain.invoke({"input": user_input})

        # 2. Prepare the input dictionary for the main chain
        chain_input = {
            "input": user_input,
            "chat_history": chat_history,
            "summary_context": summary_of_pdf,
            "topic": topic  # Add the determined topic to the input
        }

        # 3. Announce which chain is being used based on the topic
        if "admission" in topic.lower():
            print("\n[INFO] Routing to: RAG Chain (Querying PDF documents...)]")
        else:
            print("\n[INFO] Routing to: General Chain (Using summary context...)]")

        # 4. Invoke the branch with the prepared input to get the response
        response = self.branch.invoke(chain_input)

        # 5. Save the interaction to memory
        self.memory_manager.add_conversation(user_input, response)

        return response

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Ensure you have a PDF file named "Admission_process.pdf" in the same directory
    pdf_file_path = "Admission_process.pdf"
    if not os.path.exists(pdf_file_path):
        # Create a dummy file if it doesn't exist to allow the script to run
        print(f"Warning: The file '{pdf_file_path}' was not found.")
        print("Creating a dummy PDF to proceed. For real answers, please provide the actual document.")
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="This is a dummy PDF for the Gandhinagar University admission process.", ln=True, align='C')
        pdf.output(pdf_file_path)

    try:
        chatbot = ConversationalRAGChain(pdf_path=pdf_file_path)
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        print("Please ensure your GOOGLE_API_KEY is set correctly as an environment variable.")
        exit()

    print("\nChatbot initialized. Type 'quit' to exit.")
    print("-" * 50)

    while True:
        user_query = input("You: ")
        if user_query.lower() in ['quit', 'exit']:
            break
        try:
            ai_response = chatbot.invoke(user_query)
            # The [INFO] message will be printed from within the invoke method
            print(f"AI: {ai_response}\n")
        except Exception as e:
            print(f"AI: I'm sorry, I encountered an error. Please try rephrasing your question. (Error: {e})\n")

    print("-" * 50)
    print("Final Memory Stats:")
    print(json.dumps(chatbot.memory_manager.get_memory_stats(), indent=2))