from pydantic import BaseModel, Field
from typing import List, Optional
import os
import pandas as pd
from langchain.chat_models import init_chat_model
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import getpass
import numpy as np
from dotenv import load_dotenv
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import faiss
from langgraph.graph import StateGraph, END
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO

PROJECT_ROOT = Path(__file__).parent.parent  # Adjust based on your file structure
FILE_PATH = PROJECT_ROOT / "src"

load_dotenv(dotenv_path=".env", override=True)

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


class ResultClassification(BaseModel):
    category: str = Field(default="Inconclusivo", description="Conta contábil presente na lista de classes conhecidas")
    explanation: str = Field(default='', description="Justificativa plausível para atribuição da conta contábil")
    confidence: float = Field(default=0, description="Nivel de confianca na classificacao", ge=0.0, le=1.0)


class Lancamento(BaseModel):
    date: str = Field(description="Data da transação")  #TODO: include the possibility of regular payments
    desc: str = Field(description="Descrição da transação contábil")
    value: float = Field(description="Valor da transação")
    category: ResultClassification = Field(
        default=None,
        description="Resultado do modelo de classificacao"
    )
    categorized_by_llm: bool = Field(
        default=False,
        description="Indica se a classificacao foi feita por KNN ou auxiliada pelo LLM"
    )
    similar_docs: List[Document] = Field(
        default=[],
        description="Documentos similares"
    )
    status: str = Field(
        default="Pendente",
        description="Indica se a Classificação já foi validade pelo usuário",
        enum=["Pendente", "Confirmado", "Alterado"]
    )


def doc2lancamento(doc: Document) -> Lancamento:
    return Lancamento(
        date=doc.metadata['date'],
        desc=doc.page_content,
        value=doc.metadata['value']
    )


def row2doc(row_tuple) -> Document:
    index, row = row_tuple
    return Document(page_content=row["Descrição da Transação"] + "; Valor: R$" + str(row["Valor"]), metadata={
        "category": row["Conta Contábil"],
        "date": row['Data'],
        "value": row['Valor']
    })


class LLMAccountant:
    def __init__(self, threshold: float = 0.8, k: int = 10):
        self.threshold = threshold
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.model = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.index = faiss.IndexFlatL2(len(self.embeddings.embed_query("test")))
        self.source_of_truth = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        self.known_classes = ["Inconclusivo"]
        self.retriever = self.source_of_truth.as_retriever(
            search_type="mmr",  # or "similarity"
            search_kwargs={
                "k": k,
            }
        )
        self.graph = StateGraph(Lancamento)
        self.graph.add_node("retrieve", self.retrieve)
        self.graph.add_node("decide_classifier", self.decide_classifier)
        self.graph.add_node("knn", self.knn_justifier)
        self.graph.add_node("llm", self.llm_classifier)

        self.graph.set_entry_point("retrieve")
        self.graph.add_edge("retrieve", "decide_classifier")
        self.graph.add_conditional_edges(
            "decide_classifier",
            self._route_split,
            {
                "knn": "knn",
                "llm": "llm"
            }
        )
        self.graph.add_edge("knn", END)
        self.graph.add_edge("llm", END)
        self.accountant = self.graph.compile()

    def add_source_of_truth(self, docs: List[Document] | Document):
        if isinstance(docs, Document):
            docs = [docs]

        docs_ids = [str(uuid4()) for _ in range(len(docs))]
        self.source_of_truth.add_documents(docs, ids=docs_ids)
        self.known_classes.extend(
            [doc.metadata['category'] for doc in docs if doc.metadata['category'] not in self.known_classes]
        )

    def retrieve(self, query: Lancamento) -> Lancamento:
        retrieve_results = self.retriever.invoke(query.desc)
        similar_docs = [doc for doc in retrieve_results]
        query.similar_docs = similar_docs
        return query
    
    def decide_classifier(self, lancamento: Lancamento) -> Lancamento:
        similar_categories = [doc.metadata['category'] for doc in lancamento.similar_docs]
        unq_classes, counts = np.unique(similar_categories, return_counts=True)
        confidence = max(counts) / len(similar_categories)
        if confidence >= self.threshold:
            lancamento.categorized_by_llm = False
            lancamento.category = ResultClassification(
                category=unq_classes[np.argmax(counts)],
                confidence=confidence
            )
            return lancamento
        else:
            lancamento.categorized_by_llm = True
            return lancamento

    def _route_split(self, lancamento: Lancamento) -> Lancamento:
        if lancamento.categorized_by_llm:
            return "llm"
        else:
            return "knn"
        
    def knn_justifier(self, lancamento: Lancamento) -> Lancamento:
        prompt_template = open(FILE_PATH / "knn_prompt.txt", "r").read()
        prompt = ChatPromptTemplate.from_template(prompt_template).invoke({
            'category': lancamento.category.category,
            'desc': lancamento.desc
            })
        explanation = self.model.invoke(prompt).content
        lancamento.category.explanation = explanation
        return lancamento
    
    def llm_classifier(self, lancamento: Lancamento) -> Lancamento:
        prompt_template = open(FILE_PATH / "llm_prompt.txt", "r").read()
        classifier = self.model.with_structured_output(ResultClassification)
        prompt = ChatPromptTemplate.from_template(prompt_template).invoke({
            "known_classes": str(self.known_classes),
            "similar_documents": "\n\n".join(
                (f"Description: {doc.page_content}\n" f"Metadata: {doc.metadata}")
                for doc in lancamento.similar_docs
            ),
            "description": lancamento.desc
        })
        response = classifier.invoke(prompt)
        if response.confidence >= self.threshold and response.category in self.known_classes:
            lancamento.category = response
            return lancamento
        else:
            lancamento.category = ResultClassification()
            return lancamento
        
    def plot_graph(self):       
        # Get the graph as PNG bytes
        graph_png = self.accountant.get_graph().draw_mermaid_png()
        
        # Convert bytes to image
        img = mpimg.imread(BytesIO(graph_png))
        
        # Display using matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("LLM Accountant Workflow")
        plt.tight_layout()
        plt.show()

        return graph_png

    def invoke(self, lancamento: Lancamento) -> Lancamento:
        return self.accountant.invoke(lancamento)
    
    def batch(self, lancamentos: List[Lancamento]) -> List[Lancamento]:
        return self.accountant.batch(lancamentos)


if __name__ == "__main__":
    app = LLMAccountant()
    app.plot_graph()
    print("LLM Accountant application initialized.")