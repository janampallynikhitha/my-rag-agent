
import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langserve import add_routes

app = FastAPI(
    title="My RAG Agent",
    version="1.0",
    description="A free, local-model RAG playground"
)

# Sample knowledge base
documents = [
    Document(
        page_content=(
            "RAG stands for Retrieval-Augmented Generation. "
            "It retrieves relevant information from a knowledge base "
            "and uses a language model to answer questions."
        ),
        metadata={"source": "rag_basics.txt"}
    ),
    Document(
        page_content=(
            "ChromaDB stores document embeddings and retrieves "
            "relevant text using semantic similarity."
        ),
        metadata={"source": "chromadb.txt"}
    ),
    Document(
        page_content=(
            "LangServe deploys LangChain runnable applications "
            "as APIs and provides a playground interface."
        ),
        metadata={"source": "langserve.txt"}
    ),
    Document(
        page_content=(
            "Google Colab is a browser-based notebook for running Python."
        ),
        metadata={"source": "colab.txt"}
    )
]

# Build the document retrieval system
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="my_rag_documents"
)

# Load the free language model
generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

def rag_response(data: dict) -> dict:
    question = data.get("input", "").strip()

    if not question:
        return {"output": "Please enter a question."}

    docs = vectorstore.similarity_search(question, k=2)

    context = "\n".join(doc.page_content for doc in docs)

    prompt = (
        "Question: " + question + "\n"
        "Relevant information: " + context + "\n"
        "Answer:"
    )

    result = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    generated = result[0]["generated_text"]

    # Return only the generated continuation when available
    answer = generated[len(prompt):].strip()

    sources = ", ".join(
        doc.metadata.get("source", "Unknown") for doc in docs
    )

    return {
        "output": (
            f"Answer: {answer or 'No answer generated.'}\n\n"
            f"Sources: {sources}"
        )
    }

rag_chain = RunnableLambda(rag_response)

@app.get("/")
def home():
    return RedirectResponse(url="/rag/playground/")

add_routes(app, rag_chain, path="/rag")
