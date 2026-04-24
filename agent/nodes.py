from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.router import MemoryRouter
from agent.state import AgentState, default_token_budget, TokenBudget
from memory import EpisodicMemory, LongTermMemory, SemanticMemory, ShortTermMemory


@dataclass
class ContextBlock:
	label: str
	priority: int
	content: str


class AgentNodes:
	"""LangGraph nodes orchestrating routing, retrieval, response and persistence."""

	def __init__(
		self,
		short_term: ShortTermMemory,
		long_term: LongTermMemory,
		episodic: EpisodicMemory,
		semantic: SemanticMemory,
		router: MemoryRouter,
		token_limit: int = 900,
		response_token_reserve: int = 180,
		recent_turn_limit: int = 6,
	) -> None:
		self.short_term = short_term
		self.long_term = long_term
		self.episodic = episodic
		self.semantic = semantic
		self.router = router

		self.token_limit = token_limit
		self.response_token_reserve = response_token_reserve
		self.recent_turn_limit = recent_turn_limit

		# Khởi tạo LLM cho việc sinh câu trả lời
		self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

		self.system_instruction = (
			"Bạn là trợ lý AI thông minh được kết nối với hệ thống Multi-Memory. "
			"Dưới đây là các ngữ cảnh được trích xuất từ bộ nhớ của bạn theo mức độ ưu tiên:\n"
			"- [P1]: Chỉ thị hệ thống.\n"
			"- [P2]: Ngữ cảnh trò chuyện gần đây.\n"
			"- [P3]: Các đoạn ký ức được truy xuất (Sở thích, Lịch sử, Kiến thức).\n"
			"- [P4]: Lịch sử trò chuyện cũ hơn.\n\n"
			"Hãy luôn sử dụng thông tin từ bộ nhớ để trả lời người dùng một cách tự nhiên, "
			"ngắn gọn, và trực tiếp. Nếu thông tin mâu thuẫn, hãy ưu tiên thông tin mới nhất."
		)

	def _estimate_tokens(self, text: str) -> int:
		"""Sử dụng tiktoken để đếm token chuẩn xác thay vì đếm chữ."""
		if not text.strip():
			return 0
		try:
			encoding = tiktoken.encoding_for_model("gpt-4o-mini")
			return len(encoding.encode(text))
		except Exception:
			# Fallback trong trường hợp không load được tiktoken
			units = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
			return max(1, int(len(units) * 1.1))

	def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
		if max_tokens <= 0:
			return ""

		# Đoạn này vẫn dùng split() để cắt chuỗi tạm, nhưng token đếm chuẩn hơn
		words = text.split()
		if len(words) <= max_tokens:
			return text

		truncated = " ".join(words[:max_tokens]).strip()
		if not truncated.endswith("..."):
			truncated += " ..."
		return truncated

	def _format_turns(self, turns: List[Tuple[str, str]]) -> str:
		lines: List[str] = []
		for user_message, assistant_message in turns:
			lines.append(f"User: {user_message}")
			lines.append(f"Assistant: {assistant_message}")
		return "\n".join(lines)

	def _auto_trim(self, blocks: List[ContextBlock]) -> tuple[str, TokenBudget]:
		budget_limit = max(64, self.token_limit - self.response_token_reserve)
		budget = default_token_budget()
		budget["budget_limit"] = budget_limit
		used_tokens = 0
		kept_sections: List[str] = []

		# Cắt tỉa theo mức độ ưu tiên từ P1 đến P4
		for priority in [1, 2, 3, 4]:
			for block in [item for item in blocks if item.priority == priority and item.content.strip()]:
				block_tokens = self._estimate_tokens(block.content)
				remaining = budget_limit - used_tokens

				if remaining <= 0:
					budget["dropped_tokens"] += block_tokens
					continue

				if block_tokens <= remaining:
					kept_content = block.content
					kept_tokens = block_tokens
				else:
					if priority <= 3 and remaining >= 12:
						kept_content = self._truncate_to_tokens(block.content, remaining)
						kept_tokens = self._estimate_tokens(kept_content)
						budget["dropped_tokens"] += max(0, block_tokens - kept_tokens)
					else:
						budget["dropped_tokens"] += block_tokens
						continue

				used_tokens += kept_tokens
				key = f"priority_{priority}_kept"
				budget[key] += kept_tokens
				kept_sections.append(f"[{block.label}|P{priority}]\n{kept_content}")

		return "\n\n".join(kept_sections), budget

	def route_query(self, state: AgentState) -> Dict[str, Any]:
		route = self.router.route(state.get("query", ""))
		return {
			"route": route.name,
			"route_reason": route.reason,
			"route_confidence": route.confidence,
		}

	def retrieve_context(self, state: AgentState) -> Dict[str, Any]:
		user_id = state.get("user_id", "default")
		query = (state.get("query", ""))
		route = state.get("route", "short_term")

		all_recent_turns = self.short_term.get_recent_turns(limit=max(20, self.recent_turn_limit))
		recent_turns = all_recent_turns[-self.recent_turn_limit :]
		older_turns = all_recent_turns[:-self.recent_turn_limit]

		long_term_items: List[str] = []
		episodic_items: List[str] = []
		semantic_items: List[str] = []

		if route == "long_term":
			long_term_items = self.long_term.search_preferences(user_id=user_id, query=query, limit=4)
		elif route == "episodic":
			episodes = self.episodic.search(user_id=user_id, query=query, limit=4)
			episodic_items = [
				f"Q: {episode.get('query', '')} | A: {episode.get('response', '')}" for episode in episodes
			]
		elif route == "semantic":
			semantic_hits = self.semantic.search(query=query, limit=4)
			semantic_items = [hit["text"] for hit in semantic_hits if hit.get("text")]

		routed_memory_lines = long_term_items or episodic_items or semantic_items

		blocks = [
			ContextBlock(label="SYSTEM", priority=1, content=self.system_instruction),
			ContextBlock(
				label="RECENT_CONTEXT",
				priority=2,
				content=(
					f"Current user query:\n{query}\n\n"
					f"Recent turns:\n{self._format_turns(recent_turns) if recent_turns else 'No recent turns.'}"
				),
			),
			ContextBlock(
				label="ROUTED_MEMORY",
				priority=3,
				content=(
					"Retrieved memory snippets:\n"
					+ ("\n".join(f"- {item}" for item in routed_memory_lines) if routed_memory_lines else "No hits.")
				),
			),
			ContextBlock(
				label="OLDER_CONTEXT",
				priority=4,
				content=self._format_turns(older_turns),
			),
		]

		trimmed_context, token_budget = self._auto_trim(blocks)
		prompt_tokens = self._estimate_tokens(trimmed_context)

		return {
			"context_text": trimmed_context,
			"short_term_turns": [
				{"user": user_message, "assistant": assistant_message}
				for user_message, assistant_message in recent_turns
			],
			"long_term_items": long_term_items,
			"episodic_items": episodic_items,
			"semantic_items": semantic_items,
			"memory_hit": bool(routed_memory_lines),
			"token_budget": token_budget,
			"token_usage": {
				"prompt_tokens": prompt_tokens,
				"response_tokens": 0,
				"total_tokens": prompt_tokens,
			},
		}

	def generate_response(self, state: AgentState) -> Dict[str, Any]:
		"""Sử dụng trực tiếp LLM để sinh câu trả lời tự nhiên thay vì Hardcode."""
		query = (state.get("query", ""))
		context_text = state.get("context_text", "")

		# Ép LLM trả lời dựa trên ngữ cảnh đã gom và trim từ 4 loại memory
		messages = [
			SystemMessage(content=context_text),
			HumanMessage(content=query)
		]

		# Gọi LLM
		ai_msg = self.llm.invoke(messages)
		response = str(ai_msg.content)

		# Tính toán token usage chính xác
		token_usage = dict(state.get("token_usage", {}))
		prompt_tokens = token_usage.get("prompt_tokens", 0)
		response_tokens = self._estimate_tokens(response)

		# Ưu tiên lấy token đếm chuẩn từ siêu dữ liệu API nếu có
		if hasattr(ai_msg, "usage_metadata") and ai_msg.usage_metadata:
			prompt_tokens = ai_msg.usage_metadata.get("input_tokens", prompt_tokens)
			response_tokens = ai_msg.usage_metadata.get("output_tokens", response_tokens)

		token_usage["prompt_tokens"] = prompt_tokens
		token_usage["response_tokens"] = response_tokens
		token_usage["total_tokens"] = prompt_tokens + response_tokens

		return {
			"response": response,
			"token_usage": token_usage,
		}

	def _extract_semantic_fact(self, query: str) -> str:
		patterns = [
			r"ghi\s+nh[ớo]\s+ki[ếe]n\s+th[ứu]c\s*:\s*(?P<fact>.+)",
			r"remember\s+fact\s*:\s*(?P<fact>.+)",
		]

		for pattern in patterns:
			match = re.search(pattern, query, flags=re.IGNORECASE)
			if match:
				return match.group("fact").strip(" .,!?")
		return ""

	def persist_memory(self, state: AgentState) -> Dict[str, Any]:
		user_id = state.get("user_id", "default")
		query = (state.get("query", ""))
		response = (state.get("query", ""))
		route = state.get("route", "short_term")

		stored_preferences = self.long_term.update_from_user_message(user_id=user_id, message=query)
		
		semantic_fact = self._extract_semantic_fact(query)
		if semantic_fact:
			doc_id = f"user_fact_{user_id}_{abs(hash(semantic_fact))}"
			self.semantic.add_documents(
				[
					{
						"id": doc_id,
						"text": semantic_fact,
						"metadata": {"source": "user_fact", "user_id": user_id},
					}
				]
			)

		self.short_term.add_turn(user_message=query, assistant_message=response)
		self.episodic.append_episode(
			user_id=user_id,
			query=query,
			response=response,
			route=route,
			metadata={
				"memory_hit": state.get("memory_hit", False),
				"token_usage": state.get("token_usage", {}),
				"route_reason": state.get("route_reason", ""),
			},
		)

		return {
			"stored_preferences": stored_preferences,
			"ingested_semantic_fact": semantic_fact,
		}