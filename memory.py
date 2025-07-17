import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyD8CA7_kie_Q5J67e2Rig0jGXQryZgy6KI"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


class MemoryManager:
    """Advanced memory management system with short-term, long-term, and episodic memory"""
    
    def __init__(self, persistence_file: str = "memory.json"):
        self.short_term_memory = ConversationBufferMemory(k=5)
        self.long_term_memory = []
        self.episodic_memory = {}  # Key: session_id, Value: list of conversations
        self.current_session = str(uuid.uuid4())
        self.persistence_file = persistence_file
        self.episodic_memory[self.current_session] = []
        self.load_memory()
        
    def add_conversation(self, user_input: str, ai_response: str):
        """Add a conversation to all memory systems"""
        timestamp = datetime.now().isoformat()
        conversation = {
            "user_input": user_input,
            "ai_response": ai_response,
            "timestamp": timestamp
        }

        # Add to short-term memory (LangChain built-in)
        self.short_term_memory.save_context(
            {"input": user_input},
            {"output": ai_response}
        )

        # Add to long-term memory (list) with deduplication
        if not self._is_duplicated(conversation):
            self.long_term_memory.append(conversation)
            self.episodic_memory[self.current_session].append(conversation)

            # Enforce long-term memory size limit
            if len(self.long_term_memory) > 50:
                self.long_term_memory.pop(0)
        
        self.save_memory()
    
    def _is_duplicated(self, new_conversation: Dict) -> bool:
        """Check if a conversation is similar to existing ones in long-term memory"""
        if not self.long_term_memory:
            return False
        
        # Compare with last 5 conversations for similarity
        recent_conversations = self.long_term_memory[-5:]
        similarity_threshold = 0.9

        model = SentenceTransformer('all-MiniLM-L6-v2')
        new_text = f"{new_conversation['user_input']} {new_conversation['ai_response']}"
        new_embedding = model.encode(new_text)

        for conv in recent_conversations:
            existing_text = f"{conv['user_input']} {conv['ai_response']}"
            existing_embedding = model.encode(existing_text)
            similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]
            if similarity > similarity_threshold:
                return True
        return False
    
    def get_context(self, query: str, num_conversations: int = 3) -> str:
        """Retrieve relevant conversation history for context"""
        if not self.long_term_memory:
            return ""

        # Get the most recent conversations
        recent = [f"User: {conv['user_input']}\nAI: {conv['ai_response']}" 
                 for conv in self.long_term_memory[-num_conversations:]]

        # Get similar conversations based on query
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query)

        similarities = []
        for conv in self.long_term_memory:
            text = f"{conv['user_input']} {conv['ai_response']}"
            embedding = model.encode(text)
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((similarity, conv))

        # Get top 3 most similar conversations
        similarities.sort(reverse=True, key=lambda x: x[0])
        similar = [f"User: {conv['user_input']}\nAI: {conv['ai_response']}" 
                  for _, conv in similarities[:3]]

        # Combine recent and similar conversations
        context = "Recent Conversations:\n" + "\n\n".join(recent)
        if similar:
            context += "\n\nSimilar Conversations:\n" + "\n\n".join(similar)
        return context
    
    def save_memory(self):
        """Save memory to persistent storage"""
        memory_data = {
            "long_term_memory": self.long_term_memory,
            "episodic_memory": self.episodic_memory,
            "current_session": self.current_session
        }
        with open(self.persistence_file, 'w') as f:
            json.dump(memory_data, f)

    def load_memory(self):
        """Load memory from persistent storage"""
        if os.path.exists(self.persistence_file):
            with open(self.persistence_file, 'r') as f:
                memory_data = json.load(f)
            self.long_term_memory = memory_data.get("long_term_memory", [])
            self.episodic_memory = memory_data.get("episodic_memory", {})
            self.current_session = memory_data.get("current_session", str(uuid.uuid4()))

            # Initialize the current session if it doesn't exist
            if self.current_session not in self.episodic_memory:
                self.episodic_memory[self.current_session] = []

    def start_new_session(self):
        """Start a new conversation session"""
        self.current_session = str(uuid.uuid4())
        self.episodic_memory[self.current_session] = []
        self.save_memory()

    def get_memory_stats(self) -> Dict:
        """Get statistics about memory usage"""
        history = self.short_term_memory.load_memory_variables({}).get('history', '')
        return {
            "short_term_memory_count": len(history.split('\n')) // 2 if history else 0,
            "long_term_memory_count": len(self.long_term_memory),
            "episodic_memory_sessions": len(self.episodic_memory),
            "current_session_conversations": len(self.episodic_memory.get(self.current_session, [])),
            "total_conversations": sum(len(v) for v in self.episodic_memory.values())
        }

    def export_conversations(self, format: str = "json", file_path: str = None) -> str:
        """Export conversations to file"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"conversation_export_{timestamp}.{format}"
            
        if format == "json":
            with open(file_path, 'w') as f:
                json.dump({
                    "long_term_memory": self.long_term_memory,
                    "episodic_memory": self.episodic_memory
                }, f, indent=2)
        elif format == "csv":
            # Flatten all conversations
            all_conversations = []
            for session, convs in self.episodic_memory.items():
                for conv in convs:
                    all_conversations.append({
                        "session_id": session,
                        "timestamp": conv["timestamp"],
                        "user_input": conv["user_input"],
                        "ai_response": conv["ai_response"]
                    })
            df = pd.DataFrame(all_conversations)
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return file_path

    def analyze_conversation_patterns(self) -> Dict:
        """Analyze conversation patterns and topics"""
        if not self.long_term_memory:
            return {"error": "No conversations to analyze"}
            
        # Basic analysis - can be enhanced with more sophisticated NLP
        all_inputs = [conv["user_input"] for conv in self.long_term_memory]
        all_responses = [conv["ai_response"] for conv in self.long_term_memory]
        
        # Calculate average lengths
        avg_input_len = sum(len(i) for i in all_inputs) / len(all_inputs)
        avg_response_len = sum(len(r) for r in all_responses) / len(all_responses)
        
        # Count question types (simple heuristic)
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        question_counts = {word: 0 for word in question_words}
        
        for inp in all_inputs:
            lower_inp = inp.lower()
            for word in question_words:
                if lower_inp.startswith(word):
                    question_counts[word] += 1
                    
        return {
            "total_conversations": len(self.long_term_memory),
            "average_input_length": avg_input_len,
            "average_response_length": avg_response_len,
            "question_type_counts": question_counts,
            "most_common_questions": sorted(question_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }

class TerminalChatBot:
    """Terminal-based chatbot that uses the original MemoryManager class exactly as provided"""
    
    def __init__(self):
        # Initialize your original MemoryManager without modifications
        self.memory = MemoryManager()
        
        # Initialize the LLM using your original configuration
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
    def chat(self):
        """Run the terminal chat interface using your original memory system"""
        print("Terminal Chatbot (type 'quit' to exit, 'stats' for memory info)")
        print("-------------------------------------------")
        
        while True:
            try:
                user_input = input("You: ")
                
                if user_input.lower() == 'quit':
                    break
                    
                if user_input.lower() == 'stats':
                    stats = self.memory.get_memory_stats()
                    print("\nMemory Statistics:")
                    print(f"- Short-term memory: {stats['short_term_memory_count']} exchanges")
                    print(f"- Long-term memory: {stats['long_term_memory_count']} conversations")
                    print(f"- Total sessions: {stats['episodic_memory_sessions']}")
                    print(f"- Current session: {stats['current_session_conversations']} conversations\n")
                    continue
                    
                if user_input.lower() == 'new session':
                    self.memory.start_new_session()
                    print("\nStarted new conversation session\n")
                    continue
                    
                # Get context using your original get_context method
                context = self.memory.get_context(user_input)
                
                # Generate response using your original memory system
                response = self.llm.invoke(
                    f"Context from previous conversations:\n{context}\n\n"
                    f"Current question: {user_input}\n\n"
                    "Provide a helpful response in short:"
                )
                
                # Get the text response
                ai_response = response.content
                
                # Add to memory using your original add_conversation method
                self.memory.add_conversation(user_input, ai_response)
                
                print(f"\nAI: {ai_response}\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    bot = TerminalChatBot()
    bot.chat()