from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import chromadb


def _tokenize(text: str) -> List[str]:
	return re.findall(r"\w+", text.lower())


class HashEmbeddingFunction:
	"""Local deterministic embedding to keep the lab runnable offline."""

	def __init__(self, dimensions: int = 256) -> None:
		self.dimensions = dimensions

	def __call__(self, texts: List[str]) -> List[List[float]]:
		vectors: List[List[float]] = []

		for text in texts:
			vector = [0.0] * self.dimensions
			tokens = _tokenize(text)

			for token in tokens:
				digest = hashlib.md5(token.encode("utf-8")).hexdigest()
				index = int(digest, 16) % self.dimensions
				vector[index] += 1.0

			norm = math.sqrt(sum(value * value for value in vector))
			if norm > 0:
				vector = [value / norm for value in vector]

			vectors.append(vector)

		return vectors


class SemanticMemory:
	"""Semantic memory powered by Chroma persistent collection."""

	def __init__(
		self,
		persist_directory: str = "data/chroma",
		collection_name: str = "semantic_knowledge",
	) -> None:
		Path(persist_directory).mkdir(parents=True, exist_ok=True)
		self._client = chromadb.PersistentClient(path=persist_directory)
		self._collection = self._client.get_or_create_collection(
			name=collection_name,
			embedding_function=HashEmbeddingFunction(),
		)

	def count(self) -> int:
		return self._collection.count()

	def add_documents(self, documents: List[Dict[str, Any]]) -> None:
		if not documents:
			return

		ids = [str(document["id"]) for document in documents]
		texts = [str(document["text"]) for document in documents]
		metadatas = [dict(document.get("metadata", {})) for document in documents]
		self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

	def seed_default_knowledge(self) -> None:
		if self.count() > 0:
			return

		self.add_documents(
			[
				{
					"id": "k1",
					"text": "LangGraph la framework de xay dung workflow agent dang graph voi state ro rang.",
					"metadata": {"topic": "langgraph"},
				},
				{
					"id": "k2",
					"text": "Redis phu hop cho luu tru key-value toc do cao va bo nho dai han cua chatbot.",
					"metadata": {"topic": "redis"},
				},
				{
					"id": "k3",
					"text": "Ha Noi la thu do cua Viet Nam.",
					"metadata": {"topic": "geography"},
				},
				{
					"id": "k4",
					"text": "Chroma la vector database nguon mo dung de semantic search va retrieval.",
					"metadata": {"topic": "chroma"},
				},
				{
					"id": "k5",
					"text": "Token budget breakdown thuong gom system context, recent turns, memory snippets va dropped tokens.",
					"metadata": {"topic": "token_budget"},
				},
				{
					"id": "k6",
					"text": "Context window quan ly bang cach uu tien thong tin quan trong va cat bo nguu canh it gia tri.",
					"metadata": {"topic": "context_window"},
				},
				{
					"id": "k7",
					"text": "Python la ngon ngu lap trinh thong dung cho AI va automation.",
					"metadata": {"topic": "python"},
				},
				{
					"id": "k8",
					"text": "Response relevance danh gia muc do cau tra loi khop voi y dinh va keyword mong doi.",
					"metadata": {"topic": "benchmark"},
				},
			]
		)

	def search(self, query: str, limit: int = 4) -> List[Dict[str, Any]]:
		result = self._collection.query(
			query_texts=[query],
			n_results=max(1, limit),
			include=["documents", "metadatas", "distances"],
		)

		documents = result.get("documents", [[]])[0]
		metadatas = result.get("metadatas", [[]])[0]
		distances = result.get("distances", [[]])[0]

		matches: List[Dict[str, Any]] = []
		for text, metadata, distance in zip(documents, metadatas, distances):
			matches.append(
				{
					"text": text,
					"metadata": metadata or {},
					"distance": float(distance),
				}
			)

		return matches

