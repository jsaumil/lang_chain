import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# LangChain and related imports
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


# Other libraries for Memory Manager
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Environment Setup ---
# Make sure to replace this with your actual key or set it as an environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  or "AIzaSyB59PCN6s5-fLHAEF0xZP7btWRxOGt59y0"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


summary_of_pdf = """A comprehensive guide to the admission process at Gandhinagar University outlines a variety of programs, each with specific eligibility criteria, fee structures, and admission procedures. The university offers a broad spectrum of undergraduate, postgraduate, diploma, and certificate courses across several disciplines.

### **Engineering Programs**

The university offers a 4-year Bachelor of Technology (B.Tech) program with a tuition fee of Rs. 31,000 per semester. Admission is handled by the Admission Committee for Professional Courses (ACPC), and applicants must have completed their 10+2 with Physics, Chemistry, and Mathematics (PCM) and passed the GUJCET[cite: 6]. Specializations include Computer Engineering (300 seats), Information Technology (180 seats), Mechanical Engineering (180 seats), Civil Engineering (90 seats), and several others with varying seat capacities.

For postgraduate studies, a 2-year Master of Technology (M.Tech) program is available for those with a B.E. or B.Tech in a relevant field[cite: 17, 18]. The admission process is also through the ACPC, with tuition fees of Rs. 42,500 per semester[cite: 19, 21]. Specializations include Mechanical Engineering (Thermal and CAD/CAM) with 9 seats each, and Computer Engineering (Software Engineering) with 18 seats.

### **Business Administration Programs**

The Bachelor of Business Administration (BBA) is a 3-year program requiring a 10+2 qualification with English and Accountancy. Students without Accountancy in their 12th grade can take a bridge course for an additional fee of Rs. 2,000. Admissions are merit-based and handled by the university's admission committee. The program has a total intake of 700 students and a tuition fee of Rs. 35,000 per semester.

The 2-year Master of Business Administration (MBA) program is open to graduates who have taken entrance exams like CMAT, MAT, XAT, or CAT, although students without these exams can also apply[cite: 40, 41]. The ACPC manages admissions for the 180 available seats[cite: 41, 43]. The tuition fee is Rs. 41,475 per semester. Specializations include Information Technology, Marketing, Human Resources, and Finance, each with 30 seats, among others.

### **Science and Computer Science Programs**

The university offers a 3-year Bachelor of Science (B.Sc.) with various specializations, and a 4-year B.Sc. (https://www.google.com/search?q=Honours) program. Both have a tuition fee of Rs. 17,500 per semester and admissions are merit-based through the university's committee[cite: 57, 58, 73, 74]. The B.Sc. program has an intake of 480 students, while the https://www.google.com/search?q=Honours program has 240.

Master of Science (M.Sc.) programs are available in 1-year and 2-year formats, with tuition fees of Rs. 20,000 per semester. Eligibility for the 1-year program requires a 4-year B.Sc. (https://www.google.com/search?q=Hons.), while the 2-year program requires a standard B.Sc.[cite: 88, 100]. A 5-year integrated M.Sc. in Information Technology is also offered.

In computer science, the university provides 3-year programs in B.Sc. (Computer Science & Applications) and Bachelor of Computer Applications (BCA), along with a 4-year BCA (https://www.google.com/search?q=Honours). All these programs have a tuition fee of Rs. 17,500 per semester and merit-based admission through the university.

### **Other Programs**

The university also offers programs in:

  * **Medical Laboratory:** A 2-year Diploma in Medical Laboratory Technician (DMLT) and a 1-year Post Graduate Diploma (PGDMLT) are available.
  * **Commerce:** 3-year B.Com. and 4-year B.Com. (https://www.google.com/search?q=Honours) programs are offered with a tuition fee of Rs. 15,000 per semester.
  * **Law:** The university provides a 3-year LL.B., a 5-year B.A., LL.B., and a 5-year B.B.A., LL.B., all with a tuition fee of Rs. 30,000 per semester. Various 1-year law diploma programs are also available for Rs. 35,000 per .
  * **Arts:** 3-year B.A. and 4-year B.A. (https://www.google.com/search?q=Honours) programs are offered at Rs. 9,000 per semester. A 2-year M.A. program is also available for Rs. 10,000 per semester.
  * **Nursing:** The university offers a 2-year Auxiliary Nursing Midwifery (ANM) course, a 3-year General Nursing Midwifery (GNM) course, and a 4-year B.Sc. in Nursing. The medium of instruction for ANM is Gujarati, while for GNM and B.Sc. Nursing, it is English.

### **Important Admission Information**

Admissions are conducted either by the ACPC for professional courses or by the Gandhinagar University Admission Committee for most other programs, primarily based on merit[cite: 328, 329]. All stated fees are per semester unless specified otherwise and are subject to change. Applicants must have completed their previous education from a recognized board or university, and some programs have specific subject and age requirements"""



# ==============================================================================
# 1. MEMORY MANAGER CLASS (FROM YOUR FIRST SCRIPT)
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

        # Add to short-term memory
        self.short_term_memory.save_context({"input": user_input}, {"output": ai_response})

        # Add to long-term and episodic memory with deduplication
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
            return "No history yet."

        # Combine recent and semantically similar conversations
        recent = self.episodic_memory.get(self.current_session, [])[-num_conversations:]
        
        # Format the context string
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
# 2. MERGED CONVERSATIONAL CHAIN CLASS
# ==============================================================================

class ConversationalRAGChain:
    def __init__(self, pdf_path: str = "Admission_process.pdf"):
        # Initialize the memory manager
        self.memory_manager = MemoryManager()

        # Setup LLM, embeddings, and parser
        self.model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.7)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.parser = StrOutputParser()

        # Setup RAG from the provided PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

        # --- Define Chains (with memory integration) ---

        # 1. Classifier Chain (unchanged)
        classifier_prompt = PromptTemplate(
            template='Is this about Gandhinagar University admission? Answer "admission" or "general":\n{input}',
            input_variables=['input'],
        )
        self.classifier_chain = classifier_prompt | self.model | self.parser

        # 2. RAG Chain (now includes chat_history)
        rag_prompt = PromptTemplate(
            template="""You are an expert Gandhinagar University admissions counselor.
Answer the user's question based *only* on the provided context from documents and the conversation history.
If the answer is not in the context, say so.

CONVERSATION HISTORY:
{chat_history}

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {input}

ANSWER:""",
            input_variables=['chat_history', 'context', 'input'],
        )
        
        self.rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_docs(self.retriever.invoke(x["input"]))
            )
            | rag_prompt
            | self.model
            | self.parser
        )

        # 3. General Chain (now includes chat_history)
        general_prompt = PromptTemplate(
            template="""You are an expert Gandhinagar University admissions counselor. Your primary goal is to provide accurate information based on the official summary provided.

**Primary Information Source (Summary):**
Use the following summary of the Gandhinagar University Admission Process Guide to answer the user's question. All facts, figures, and admission details must come from this source.

SUMMARY CONTEXT:
{summmary_context}

**Conversation Context:**
Use the following conversation history to understand the user's previous questions and the flow of dialogue.

CONVERSATION HISTORY:
{chat_history}

**Instructions:**
1. Answer the user's question using the context of information from the **SUMMARY CONTEXT**.
2. if the question is not answered in the **SUMMARY CONTEXT**, then answer by your knowledge. but answer like a Gandhinagar University admissions counselor
3. Refer to the **CONVERSATION HISTORY** to understand the context of the question, but do not use it as a source for factual answers.

**QUESTION:**
{input}


ANSWER:""",
            input_variables=['chat_history', 'input','summmary_context'],
        )
        self.general_chain = general_prompt | self.model | self.parser

        self.branch_chain = RunnableBranch(
            (RunnableLambda(lambda x: "admission" in x["input"].lower(), self.rag_chain),
            RunnableLambda(lambda x: "general" in x["input"].lower(), self.general_chain),
            )
        )
    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    # MODIFIED invoke() method

def invoke(self, user_input: str) -> str:
    """
    The main method now prepares inputs and calls the pre-built chain.
    """
    # 1. Get relevant context from memory
    chat_history = self.memory_manager.get_context(user_input)
    
    # 2. Prepare the input dictionary for the chain
    chain_input = {
        "input": user_input,
        "chat_history": chat_history,
        "summary_context": summary_of_pdf
    }
    
    # 3. Invoke the main chain (this single line replaced the entire if/else block)
    response = self.full_chain.invoke(chain_input)
    
    # 4. Add the new interaction to memory
    self.memory_manager.add_conversation(user_input, response)
    
    return response



# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Create an instance of our conversational chain
    # Ensure you have a PDF file named "Admission_process.pdf" in the same directory
    try:
        chatbot = ConversationalRAGChain(pdf_path="Admission_process.pdf")
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        print("Please ensure 'Admission_process.pdf' exists and your GOOGLE_API_KEY is correct.")
        exit()

    print("Chatbot initialized. Type 'quit' to exit.")
    print("-" * 50)

    # # First query
    # query1 = "What documents are needed for M Tech admission?"
    # print(f"User: {query1}")
    # response1 = chatbot.invoke(query1)
    # print(f"AI: {response1}\n")

    # # Second query that should use the memory of the first one
    # query2 = "what is the fee structure for that program?"
    # print(f"User: {query2}")
    # response2 = chatbot.invoke(query2)
    # print(f"AI: {response2}\n")

    # # Third query (general knowledge)
    # query3 = "What is a fun fact about computers?"
    # print(f"User: {query3}")
    # response3 = chatbot.invoke(query3)
    # print(f"AI: {response3}\n")
    
    # You can now interact with it in a loop
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'quit':
            break
        ai_response = chatbot.invoke(user_query)
        print(f"AI: {ai_response}\n")

    print("-" * 50)
    print("Final Memory Stats:")
    print(json.dumps(chatbot.memory_manager.get_memory_stats(), indent=2))