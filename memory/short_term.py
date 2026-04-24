from __future__ import annotations

from typing import List, Tuple

from langchain.memory import ConversationBufferMemory


class ShortTermMemory:
	"""Short-term conversation memory powered by ConversationBufferMemory."""

	def __init__(self) -> None:
		self._memory = ConversationBufferMemory(
			memory_key="history",
			input_key="input",
			output_key="output",
			return_messages=True,
		)

	def add_turn(self, user_message: str, assistant_message: str) -> None:
		self._memory.save_context(
			{"input": user_message},
			{"output": assistant_message},
		)

	def get_recent_turns(self, limit: int = 6) -> List[Tuple[str, str]]:
		messages = self._memory.chat_memory.messages
		turns: List[Tuple[str, str]] = []
		pending_user: str | None = None

		for message in messages:
			if message.type == "human":
				pending_user = message.content
			elif message.type == "ai" and pending_user is not None:
				turns.append((pending_user, message.content))
				pending_user = None

		return turns[-limit:]

	def get_history_text(self, limit: int = 6) -> str:
		lines: List[str] = []
		for user_message, assistant_message in self.get_recent_turns(limit=limit):
			lines.append(f"User: {user_message}")
			lines.append(f"Assistant: {assistant_message}")
		return "\n".join(lines)

	def clear(self) -> None:
		self._memory.clear()

