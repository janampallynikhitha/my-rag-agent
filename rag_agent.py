
import re

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.runnables import RunnableLambda
from langserve import add_routes
from pydantic import BaseModel

app = FastAPI(
    title="My RAG Agent",
    version="1.0",
    description="A lightweight RAG playground"
)

# Small sample knowledge base
documents = [
    {
        "text": (
            "RAG stands for Retrieval-Augmented Generation. "
            "It retrieves relevant information from a knowledge base "
            "and uses it to answer questions."
        ),
        "source": "rag_basics.txt"
    },
    {
        "text": (
            "ChromaDB stores document embeddings and retrieves "
            "relevant text using semantic similarity."
        ),
        "source": "chromadb.txt"
    },
    {
        "text": (
            "LangServe deploys LangChain runnable applications "
            "as APIs and provides a playground interface."
        ),
        "source": "langserve.txt"
    },
    {
        "text": (
            "Google Colab is a browser-based notebook for running Python."
        ),
        "source": "colab.txt"
    }
]


def tokenize(text):
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))

class RagInput(BaseModel):
    input: str
def rag_response(data: dict) -> dict:
    question = data.get("input", "").strip()

    if not question:
        return {"output": "Please enter a question."}

    question_words = tokenize(question)

    if not question_words:
        return {"output": "Please enter a valid question."}

    # Retrieve the most relevant documents without loading ML models.
    ranked = []

    for doc in documents:
        doc_words = tokenize(doc["text"])
        score = len(question_words & doc_words)
        ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    relevant = [
        doc for score, doc in ranked[:2] if score > 0
    ]

    if not relevant:
        return {
            "output": (
                "I couldn't find relevant information "
                "in the current knowledge base."
            )
        }

    context = "\n".join(doc["text"] for doc in relevant)
    sources = ", ".join(doc["source"] for doc in relevant)

    return {
        "output": (
            f"Relevant information:\n{context}\n\n"
            f"Sources: {sources}"
        )
    }


rag_chain = RunnableLambda(rag_response).with_types(
    input_type=RagInput
)


@app.get("/")
def home():
    return RedirectResponse(url="/rag/playground/")


add_routes(app, rag_chain, path="/rag")

