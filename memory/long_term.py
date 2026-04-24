from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

try:
	import redis
	from redis.exceptions import RedisError
except Exception:  # pragma: no cover - graceful fallback when redis is unavailable
	redis = None

	class RedisError(Exception):
		pass


def _tokenize(text: str) -> List[str]:
	return re.findall(r"\w+", text.lower())


class LongTermMemory:
	"""Long-term memory via Redis with an in-process fallback map."""

	def __init__(
		self,
		redis_url: str = "redis://localhost:6379/0",
		namespace: str = "mma",
	) -> None:
		self._namespace = namespace
		self._local_store: Dict[str, Dict[str, str]] = defaultdict(dict)
		self._client = None

		if redis is None:
			return

		try:
			client = redis.Redis.from_url(redis_url, decode_responses=True)
			client.ping()
			self._client = client
		except RedisError:
			self._client = None

	def _redis_key(self, user_id: str) -> str:
		return f"{self._namespace}:user:{user_id}:preferences"

	def _merge_value(self, old_value: str | None, new_value: str) -> str:
		cleaned_new = new_value.strip()
		if not old_value:
			return cleaned_new

		old_parts = [part.strip() for part in old_value.split(";") if part.strip()]
		if cleaned_new in old_parts:
			return old_value
		old_parts.append(cleaned_new)
		return "; ".join(old_parts)

	def set_preference(self, user_id: str, key: str, value: str) -> None:
		if self._client is not None:
			redis_key = self._redis_key(user_id)
			old_value = self._client.hget(redis_key, key)
			merged_value = self._merge_value(old_value, value)
			self._client.hset(redis_key, mapping={key: merged_value})
			return

		old_value = self._local_store[user_id].get(key)
		self._local_store[user_id][key] = self._merge_value(old_value, value)

	def get_preferences(self, user_id: str) -> Dict[str, str]:
		if self._client is not None:
			return dict(self._client.hgetall(self._redis_key(user_id)))
		return dict(self._local_store[user_id])

	def search_preferences(self, user_id: str, query: str, limit: int = 5) -> List[str]:
		preferences = self.get_preferences(user_id)
		if not preferences:
			return []

		query_tokens = set(_tokenize(query))
		ranked: List[Tuple[int, str]] = []

		for key, value in preferences.items():
			text = f"{key}: {value}".lower()
			overlap = sum(1 for token in query_tokens if token in text)
			ranked.append((overlap, f"{key}: {value}"))

		ranked.sort(key=lambda item: item[0], reverse=True)
		matches = [text for score, text in ranked if score > 0]
		if matches:
			return matches[:limit]

		return [f"{key}: {value}" for key, value in list(preferences.items())[:limit]]

	def _infer_preference_key(self, value: str) -> str:
		normalized = value.lower()

		if any(word in normalized for word in ["coffee", "cà phê", "trà", "tea", "drink", "đồ uống"]):
			return "favorite_drink"
		if any(word in normalized for word in ["book", "sách", "read", "đọc"]):
			return "favorite_book"
		if any(word in normalized for word in ["movie", "film", "phim", "series"]):
			return "favorite_movie"
		if any(word in normalized for word in ["framework", "python", "java", "langgraph", "tool"]):
			return "favorite_tech"
		return "general_preference"

	def extract_preferences(self, message: str) -> Dict[str, str]:
		patterns = [
			r"(?:tôi|toi|mình|minh)\s+thích\s+(?P<value>.+)",
			r"(?:sở thích của tôi là|so thich cua toi la)\s+(?P<value>.+)",
			r"i\s+like\s+(?P<value>.+)",
			r"my\s+favorite\s+.+?\s+is\s+(?P<value>.+)",
			r"hãy nhớ rằng tôi thích\s+(?P<value>.+)",
			r"hay nho rang toi thich\s+(?P<value>.+)",
		]

		extracted: Dict[str, str] = {}
		for pattern in patterns:
			match = re.search(pattern, message, flags=re.IGNORECASE)
			if not match:
				continue
			value = match.group("value").strip(" .,!?")
			if not value:
				continue
			key = self._infer_preference_key(value)
			extracted[key] = value
			break

		return extracted

	def update_from_user_message(self, user_id: str, message: str) -> Dict[str, str]:
		extracted = self.extract_preferences(message)
		for key, value in extracted.items():
			self.set_preference(user_id, key, value)
		return extracted

