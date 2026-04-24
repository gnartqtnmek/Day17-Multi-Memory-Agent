from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _tokenize(text: str) -> List[str]:
	return re.findall(r"\w+", text.lower())


class EpisodicMemory:
	"""Episodic memory persisted as a JSON array on disk."""

	def __init__(self, file_path: str = "data/episodic_log.json") -> None:
		self._path = Path(file_path)
		self._path.parent.mkdir(parents=True, exist_ok=True)
		if not self._path.exists():
			self._path.write_text("[]", encoding="utf-8")

	def _read_all(self) -> List[Dict[str, Any]]:
		raw = self._path.read_text(encoding="utf-8").strip()
		if not raw:
			return []
		try:
			data = json.loads(raw)
			if isinstance(data, list):
				return data
		except json.JSONDecodeError:
			pass
		return []

	def _write_all(self, episodes: List[Dict[str, Any]]) -> None:
		self._path.write_text(
			json.dumps(episodes, ensure_ascii=False, indent=2),
			encoding="utf-8",
		)

	def append_episode(
		self,
		user_id: str,
		query: str,
		response: str,
		route: str,
		metadata: Dict[str, Any] | None = None,
	) -> None:
		episodes = self._read_all()
		episodes.append(
			{
				"timestamp": datetime.now(timezone.utc).isoformat(),
				"user_id": user_id,
				"query": query,
				"response": response,
				"route": route,
				"metadata": metadata or {},
			}
		)
		self._write_all(episodes)

	def get_recent(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
		episodes = [episode for episode in self._read_all() if episode.get("user_id") == user_id]
		return episodes[-limit:]

	def search(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
		episodes = [episode for episode in self._read_all() if episode.get("user_id") == user_id]
		query_tokens = set(_tokenize(query))

		ranked: List[tuple[int, Dict[str, Any]]] = []
		for episode in episodes:
			merged_text = f"{episode.get('query', '')} {episode.get('response', '')}".lower()
			score = sum(1 for token in query_tokens if token in merged_text)
			ranked.append((score, episode))

		ranked.sort(key=lambda item: item[0], reverse=True)
		matched = [episode for score, episode in ranked if score > 0]
		if matched:
			return matched[:limit]
		return episodes[-limit:]

