from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MemoryRoute:
	name: str
	reason: str
	confidence: float


class MemoryRouter:
	"""Intent router that chooses the most suitable memory backend."""

	def __init__(self) -> None:
		self._patterns: Dict[str, List[str]] = {
			"episodic": [
				r"l[úu]c\s+n[ãa]y",
				r"tr[ướu]c\s+[đd][óo]",
				r"h[ôo]m\s+qua",
				r"v[ừu]a\s+n[óo]i",
				r"what\s+did\s+i\s+say",
				r"earlier",
				r"history",
			],
			"long_term": [
				r"s[ởo]\s+th[íi]ch",
				r"b[ạa]n\s+nh[ớo]",
				r"h[ãa]y\s+nh[ớo]",
				r"nh[ớo]\s+r[ằa]ng",
				r"t[ôo]i\s+th[íi]ch",
				r"m[ìi]nh\s+th[íi]ch",
				r"favorite",
				r"preference",
				r"i\s+like",
				r"remember\s+that",
			],
			"semantic": [
				r"l[àa]\s+g[ìi]",
				r"t[ạa]i\s+sao",
				r"[đd][ịi]nh\s+ngh[ĩi]a",
				r"explain",
				r"define",
				r"what\s+is",
				r"compare",
				r"ki[ếe]n\s+th[ứu]c",
			],
		}
		self._tie_break_order = ["episodic", "long_term", "semantic", "short_term"]

	def route(self, query: str) -> MemoryRoute:
		normalized = query.strip().lower()
		scores = {"episodic": 0, "long_term": 0, "semantic": 0, "short_term": 0}
		reasons: List[str] = []

		for route_name, patterns in self._patterns.items():
			for pattern in patterns:
				if re.search(pattern, normalized, flags=re.IGNORECASE):
					scores[route_name] += 1
					reasons.append(f"matched:{pattern}")

		if all(score == 0 for route_name, score in scores.items() if route_name != "short_term"):
			if normalized.endswith("?"):
				scores["semantic"] = 1
				reasons.append("fallback:question_form")
			else:
				scores["short_term"] = 1
				reasons.append("fallback:default_short_term")

		best_route = "short_term"
		best_score = -1
		for route_name in self._tie_break_order:
			score = scores.get(route_name, 0)
			if score > best_score:
				best_score = score
				best_route = route_name

		confidence = min(0.95, 0.45 + 0.15 * max(1, best_score))
		reason = ", ".join(reasons[:3]) if reasons else "no_signal_default"

		return MemoryRoute(name=best_route, reason=reason, confidence=confidence)

