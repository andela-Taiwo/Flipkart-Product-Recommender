"""
RAG (Retrieval-Augmented Generation) Chain Builder

This module implements a RAG system for product recommendations using:
- Vector store for semantic search over product reviews
- LLM for generating contextual responses
- Chat history for maintaining conversation context
- History-aware retrieval for better query understanding
"""

from langchain_core.prompts import  MessagesPlaceholder, ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config
from langchain_astradb import AstraDBVectorStore


class RAGChainBuilder:
    """
    Builds and manages a RAG (Retrieval-Augmented Generation) chain for product recommendations.
    
    The RAG chain combines:
    1. Vector store retrieval for finding relevant product reviews
    2. LLM for generating natural language responses
    3. Chat history for maintaining conversation context
    4. History-aware retrieval for understanding follow-up questions
    """
    
    def __init__(self, vector_store: AstraDBVectorStore):
        """
        Initialize the RAGChainBuilder with the vector store.
        
        Args:
            vector_store: Pre-configured AstraDB vector store containing embedded product reviews
            
        Step-by-step initialization:
        1. Store the vector store reference for later retrieval
        2. Load configuration (API keys, model names, etc.)
        3. Initialize the LLM (ChatGroq) with specified model and temperature
        4. Create an empty dictionary to store chat histories per session
        """
        self.vector_store = vector_store
        self.config = Config()
        # Initialize the LLM that will generate responses
        self.llm = ChatGroq(model=self.config.LLM_MODEL, temperature=self.config.LLM_TEMPERATURE)
        # Dictionary to store chat history for each session (key: session_id, value: ChatMessageHistory)
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get or create chat history for a given session.
        
        This method is used by RunnableWithMessageHistory to manage conversation history.
        Each session (identified by session_id) maintains its own conversation history.
        
        Args:
            session_id: Unique identifier for the user session
            
        Returns:
            ChatMessageHistory instance for the session
            
        Step-by-step:
        1. Check if history exists for this session_id
        2. If not, create a new ChatMessageHistory instance
        3. Store it in history_store for future use
        4. Return the history instance
        """
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def build_rag_chain(self):
        """
        Build the complete RAG chain with history awareness.
        
        This method constructs a multi-stage RAG pipeline:
        
        Pipeline Flow:
        1. User asks a question
        2. History-aware retriever rewrites the question using chat history
        3. Retriever searches vector store for relevant product reviews
        4. QA chain combines retrieved context with the question
        5. LLM generates answer using context and history
        6. Answer is returned and stored in history
        
        Returns:
            RunnableWithMessageHistory: The complete RAG chain ready for invocation
            
        Step-by-step construction:
        """
        # STEP 1: Create a retriever from the vector store
        # The retriever will search for the top k=3 most relevant documents
        # based on semantic similarity to the user's query
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # STEP 2: Create the context rewriting prompt
        # This prompt helps the LLM rewrite user questions to be more specific
        # by considering the chat history. For example:
        # - User: "What about the battery?"
        # - History: Previous question was about a phone
        # - Rewritten: "What about the battery life of the phone?"
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given the chat history and user question, rewrite the user question to be more specific and contextually relevant to the question"""),
            MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
            ("human", "{input}"),  # Placeholder for user input
        ])
        
        # STEP 3: Create the QA (Question-Answering) prompt
        # This prompt instructs the LLM on how to answer questions using:
        # - The retrieved context (product reviews)
        # - The user's question
        # - The chat history for context
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an e-commerce bot answering product related queries using reviews and titles from the product reviews database.
            stick to the context and provide contextually relevant information in your response. Be concise and to the point: \n\n CONTEXT:\n{context}\n\nQUESTION: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
            ("human", "{input}"),  # Placeholder for user input
        ])
        
        # STEP 4: Create history-aware retriever
        # This combines the retriever with the context rewriting prompt.
        # Before searching, it uses the LLM to rewrite the question based on chat history,
        # making follow-up questions more effective.
        # Example flow:
        #   - Original query: "What about its price?"
        #   - With history: "What about the price of the Samsung Galaxy phone?"
        #   - Then searches vector store with the rewritten query
        history_aware_retriever = create_history_aware_retriever(
            self.llm, 
            retriever, 
            context_prompt
        )
        
        # STEP 5: Create the QA chain
        # This chain takes the retrieved documents and the question,
        # then uses the LLM to generate an answer based on the context.
        # The "stuff" method means it puts all retrieved documents into the prompt.
        qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # STEP 6: Combine retriever and QA chain into a retrieval chain
        # This creates the core RAG chain:
        #   1. Takes user input
        #   2. Uses history-aware retriever to find relevant documents
        #   3. Passes documents + question to QA chain
        #   4. Returns the answer
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
        
        # STEP 7: Wrap the chain with message history management
        # This adds conversation history tracking to the RAG chain:
        #   - Automatically stores user messages and responses
        #   - Provides history to prompts via MessagesPlaceholder
        #   - Manages history per session using _get_history method
        #   - Maps input/output keys correctly
        return RunnableWithMessageHistory(
            rag_chain,  # The RAG chain to wrap
            self._get_history,  # Function to get/create history for a session
            input_messages_key="input",  # Key for user input in the chain
            history_messages_key="chat_history",  # Key for history in prompts
            output_messages_key="answer",  # Key for the answer in the output
        )