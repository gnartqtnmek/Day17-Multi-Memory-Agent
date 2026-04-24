from __future__ import annotations

from typing import Dict, List, TypedDict


class TokenBudget(TypedDict):
	priority_1_kept: int
	priority_2_kept: int
	priority_3_kept: int
	priority_4_kept: int
	dropped_tokens: int
	budget_limit: int


def default_token_budget() -> TokenBudget:
	return {
		"priority_1_kept": 0,
		"priority_2_kept": 0,
		"priority_3_kept": 0,
		"priority_4_kept": 0,
		"dropped_tokens": 0,
		"budget_limit": 0,
	}


class AgentState(TypedDict, total=False):
	user_id: str
	query: str
	route: str
	route_reason: str
	route_confidence: float
	context_text: str
	response: str

	short_term_turns: List[Dict[str, str]]
	long_term_items: List[str]
	episodic_items: List[str]
	semantic_items: List[str]

	memory_hit: bool
	stored_preferences: Dict[str, str]

	token_budget: TokenBudget
	token_usage: Dict[str, int]

