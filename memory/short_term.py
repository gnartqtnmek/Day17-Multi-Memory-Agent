from __future__ import annotations

from typing import List, Tuple


class ShortTermMemory:
	"""Short-term memory management using a sliding window list.
	Không phụ thuộc vào langchain.memory để tránh lỗi import và tăng tốc độ.
	"""

	def __init__(self, window_size: int = 20) -> None:
		self.window_size = window_size
		self.history: List[Tuple[str, str]] = []

	def add_turn(self, user_message: str, assistant_message: str) -> None:
		"""Thêm một lượt hội thoại mới vào bộ nhớ ngắn hạn."""
		self.history.append((user_message, assistant_message))
		
		if len(self.history) > self.window_size:
			self.history.pop(0)

	def get_recent_turns(self, limit: int = 6) -> List[Tuple[str, str]]:
		"""Lấy danh sách các lượt hội thoại gần nhất."""
		if limit <= 0:
			return []
		return self.history[-limit:]
		
	def clear(self) -> None:
		"""Xóa toàn bộ lịch sử (dùng khi bắt đầu session mới)."""
		self.history.clear()