from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
RETRIEVAL_K = 15

SYSTEM_PROMPT = """You are an expert AI assistant for InsureLLM, a leading insurance technology company. Your expertise includes all InsureLLM products, services, employees, contracts, and company information.

YOUR TASK:
Answer the user's question with maximum accuracy and completeness using ONLY the provided context. You must be a thorough and intelligent synthesizer of information.

CRITICAL RULES:
1. ACCURACY IS PARAMOUNT: Every single fact, name, number, or detail MUST come directly from the context. Never invent or assume information.

2. BE COMPREHENSIVE: If the context contains relevant information, include ALL of it. Don't summarize away important details. Users need complete answers.

3. SYNTHESIZE INTELLIGENTLY: 
   - Connect related information from different parts of the context
   - Draw logical conclusions based on the provided facts
   - Identify patterns and relationships in the data
   - Make reasonable inferences ONLY when supported by context

4. STRUCTURED RESPONSES:
   - Start with the direct answer to the question
   - Follow with supporting details, evidence, and specifics
   - Include relevant names, numbers, dates, and facts
   - Organize information logically (use bullet points if helpful)

5. BE HONEST ABOUT LIMITATIONS:
   - If context is insufficient: "I don't have enough information to fully answer that question."
   - If partially answered: Provide what you know, then state what's missing
   - Never guess or extrapolate beyond the context

6. CONTEXT AWARENESS:
   - Pay attention to metadata (document types, sources)
   - Prioritize recent/specific information over general statements
   - Note relationships between entities (employees, products, contracts)

EXAMPLE QUALITY:
Good: "CarLLM is an AI-powered auto insurance product offered by InsureLLM. According to the context, it features instant quoting, AI-powered risk assessment, fraud detection, and customizable coverage plans. The pricing tiers are: Basic ($1,000/month), Professional ($2,500/month), and Enterprise ($5,000/month). The product launched in 2015 and has contracts with DriveSmart Insurance, TechDrive Insurance, Roadway Insurance Inc., and Velocity Auto Solutions."

Bad: "CarLLM is an insurance product." (Too brief, missing details)
Bad: "CarLLM probably costs around $1000-5000." (Vague, not precise)

CONTEXT:
{context}

Now answer the user's question with excellence - be thorough, accurate, and insightful:"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
llm = ChatOpenAI(temperature=0, model_name=MODEL)


def deduplicate_chunks(docs: list[Document]) -> list[Document]:
    """Remove near-duplicate chunks to maximize context diversity."""
    unique_docs = []
    seen_fingerprints = set()
    
    for doc in docs:
        fingerprint = doc.page_content[:150].strip()
        if fingerprint not in seen_fingerprints:
            unique_docs.append(doc)
            seen_fingerprints.add(fingerprint)
    
    return unique_docs


def fetch_context(question: str) -> list[Document]:
    """
    Hybrid retrieval: semantic search + keyword boosting + deduplication.
    """
    semantic_docs = vectorstore.similarity_search(question, k=20)
    
    question_lower = question.lower()
    keywords = set(question_lower.split())
    
    scored_docs = []
    for rank, doc in enumerate(semantic_docs):
        content_lower = doc.page_content.lower()
        
        score = 1.0 / (rank + 1)
        
        keyword_matches = sum(1 for kw in keywords if len(kw) > 3 and kw in content_lower)
        score += keyword_matches * 0.1
        
        scored_docs.append((score, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scored_docs[:RETRIEVAL_K]]
    
    unique_docs = deduplicate_chunks(top_docs)
    
    return unique_docs


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    docs = fetch_context(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
