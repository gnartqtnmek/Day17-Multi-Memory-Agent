from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Mapping, Protocol, Tuple, cast

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

try:
	import redis
except Exception:
	redis = None


def _tokenize(text: str) -> List[str]:
	return re.findall(r"\w+", text.lower())


class FactExtraction(BaseModel):
	key: str = Field(description="Loại thông tin (vd: allergy, favorite_drink, tech_stack, goal)")
	value: str = Field(description="Giá trị của thông tin (vd: đậu nành, cà phê, Python)")
	action: str = Field(description="Hành động: 'UPSERT' (lưu mới hoặc đè lên cũ) hoặc 'DELETE' (xóa bỏ thông tin)")

class FactList(BaseModel):
	facts: List[FactExtraction]


class _RedisClient(Protocol):
	def ping(self) -> object: ...

	def hgetall(self, name: str) -> Mapping[str, str] | Mapping[bytes, bytes]: ...

	def hset(self, name: str, mapping: Mapping[str, str]) -> object: ...

	def hdel(self, name: str, *keys: str) -> object: ...


class LongTermMemory:
	"""Long-term memory via Redis with an in-process fallback map, powered by LLM extraction."""

	def __init__(
		self,
		redis_url: str = "redis://localhost:6379/0",
		namespace: str = "mma",
	) -> None:
		self._namespace = namespace
		self._local_store: Dict[str, Dict[str, str]] = defaultdict(dict)
		self._client: _RedisClient | None = None

		self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(FactList)

		if redis is None:
			return

		try:
			client = cast(_RedisClient, redis.Redis.from_url(redis_url, decode_responses=True))
			client.ping()
			self._client = client
		except Exception:
			self._client = None

	def _redis_key(self, user_id: str) -> str:
		return f"{self._namespace}:user:{user_id}:preferences"

	def get_preferences(self, user_id: str) -> Dict[str, str]:
		if self._client is not None:
			raw_preferences = self._client.hgetall(self._redis_key(user_id))
			return {str(key): str(value) for key, value in raw_preferences.items()}
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

	def update_from_user_message(self, user_id: str, message: str) -> Dict[str, str]:
		"""
		Sử dụng LLM để đối chiếu ngữ cảnh hiện tại và cập nhật thông minh (Upsert/Delete).
		Thay thế hoàn toàn logic Regex cũ để xử lý được các ca khó (Conflict Handling).
		"""
		current_profile = self.get_preferences(user_id)
		if current_profile:
			current_context = "\n".join([f"- {k}: {v}" for k, v in current_profile.items()])
		else:
			current_context = "Chưa có thông tin nào."

		prompt = ChatPromptTemplate.from_messages([
			("system", 
			 "Bạn là chuyên gia trích xuất dữ liệu cá nhân từ hội thoại.\n"
			 "Profile hiện tại của user:\n{profile}\n\n"
			 "Nhiệm vụ: Phân tích câu nói của user để tìm các fact, sở thích hoặc thông tin cá nhân cần nhớ dài hạn.\n"
			 "- Nếu user cung cấp thông tin mới, HOẶC cung cấp thông tin phủ định/thay thế thông tin cũ, hãy đặt action='UPSERT'.\n"
			 "- Nếu user yêu cầu quên hoặc xóa một thông tin, hãy đặt action='DELETE'.\n"
			 "Trả về rỗng nếu câu nói không chứa thông tin cần nhớ dài hạn."
			),
			("human", "{message}")
		])

		try:
			extracted_raw = self.llm.invoke(prompt.format(profile=current_context, message=message))
			extracted_facts = FactList.model_validate(extracted_raw)
		except Exception as e:
			print(f"[LongTermMemory] LLM Extraction Error: {e}")
			return {}

		updates: Dict[str, str] = {}
		if not extracted_facts or not extracted_facts.facts:
			return updates

		for fact in extracted_facts.facts:
			if fact.action == "UPSERT":
				if self._client:
					self._client.hset(self._redis_key(user_id), mapping={fact.key: fact.value})
				else:
					self._local_store[user_id][fact.key] = fact.value
				updates[fact.key] = fact.value
				
			elif fact.action == "DELETE":
				if self._client:
					self._client.hdel(self._redis_key(user_id), fact.key)
				else:
					self._local_store[user_id].pop(fact.key, None)
				updates[fact.key] = "[DELETED]"

		return updates