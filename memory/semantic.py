from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import chromadb
import chromadb.utils.embedding_functions as embedding_functions


class SemanticMemory:
	"""Semantic memory powered by Chroma persistent collection and OpenAI Embeddings."""

	def __init__(
		self,
		persist_directory: str = "data/chroma",
		collection_name: str = "semantic_knowledge",
	) -> None:
		Path(persist_directory).mkdir(parents=True, exist_ok=True)
		self._client = chromadb.PersistentClient(path=persist_directory)

		api_key = os.environ.get("OPENAI_API_KEY")
		if not api_key:
			print("Cảnh báo: Không tìm thấy OPENAI_API_KEY. Semantic Search có thể báo lỗi.")

		embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
			api_key=api_key,
			model_name="text-embedding-3-small"
		)

		self._collection = self._client.get_or_create_collection(
			name=collection_name,
			embedding_function=cast(Any, embedding_fn), # Đã ép kiểu an toàn cho Pylance
		)

	def count(self) -> int:
		return self._collection.count()

	def add_documents(self, documents: List[Dict[str, Any]]) -> None:
		if not documents:
			return

		ids = [str(document["id"]) for document in documents]
		texts = [str(document["text"]) for document in documents]
		
		metadatas = cast(Any, [document.get("metadata", {}) for document in documents])
		
		self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

	def load_from_json(self, file_path: str = "hotpot_120.json") -> None:
		"""Hỗ trợ nạp dữ liệu động từ file JSON bên ngoài để mở rộng tri thức (RAG)."""
		path = Path(file_path)
		if not path.exists():
			return
			
		try:
			data = json.loads(path.read_text(encoding="utf-8"))
			if isinstance(data, list):
				docs = []
				for i, item in enumerate(data):
					docs.append({
						"id": item.get("id", f"doc_ext_{i}"),
						"text": item.get("text", str(item)),
						"metadata": item.get("metadata", {"source": file_path})
					})
				self.add_documents(docs)
				print(f"[SemanticMemory] Đã nạp thành công {len(docs)} tài liệu từ {file_path}")
		except Exception as e:
			print(f"[SemanticMemory] Lỗi khi nạp dữ liệu từ {file_path}: {e}")

	def seed_default_knowledge(self) -> None:
		if self.count() > 0:
			return

		lab_data = [
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
			{
				"id": "k9",
				"text": "Short-term memory trong lab nay duoc cai dat bang ConversationBufferMemory.",
				"metadata": {"topic": "short_term"},
			},
			{
				"id": "k10",
				"text": "Episodic memory duoc luu tren file JSON de ghi log theo tung turn.",
				"metadata": {"topic": "episodic"},
			},
			{
				"id": "k11",
				"text": "Memory router phan tich intent cua truy van de chon backend phu hop nhu long-term, episodic hoac semantic.",
				"metadata": {"topic": "router"},
			},
			{
				"id": "k12",
				"text": "Khi token gan cham nguong, he thong auto-trim theo uu tien P1 den P4 de toi uu context window.",
				"metadata": {"topic": "auto_trim"},
			},
		]
		self.add_documents(lab_data)
		
		self.load_from_json()

	def search(self, query: str, limit: int = 4) -> List[Dict[str, Any]]:
		result = self._collection.query(
			query_texts=[query],
			n_results=max(1, limit),
			include=["documents", "metadatas", "distances"],
		)

		raw_docs = result.get("documents")
		documents = raw_docs[0] if raw_docs else []

		raw_metas = result.get("metadatas")
		metadatas = raw_metas[0] if raw_metas else []

		raw_dists = result.get("distances")
		distances = raw_dists[0] if raw_dists else []

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