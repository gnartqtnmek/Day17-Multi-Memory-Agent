from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import tiktoken

from agent.graph import MultiMemoryAgent


def _estimate_tokens(text: str) -> int:
	if not text.strip():
		return 0
	try:
		# Sử dụng tiktoken để đếm token chuẩn xác thay vì đếm chữ
		encoding = tiktoken.encoding_for_model("gpt-4o-mini")
		return len(encoding.encode(text))
	except Exception:
		# Fallback an toàn nếu có lỗi thư viện
		units = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
		return max(1, int(len(units) * 1.1))


def _safe_divide(numerator: float, denominator: float) -> float:
	if denominator == 0:
		return 0.0
	return numerator / denominator


class NoMemoryAgent:
	"""Simple baseline that does not retrieve from any memory backend."""

	def chat(self, query: str, user_id: str = "baseline") -> Dict[str, Any]:
		lowered = query.lower()
		if "?" in lowered or lowered.strip().endswith("?"):
			response = (
				"Mình không dùng memory backend trong chế độ baseline, "
				"nên chỉ có thể phản hồi chung và cần thêm ngữ cảnh trực tiếp."
			)
		else:
			response = "Mình đã nhận thông tin, nhưng baseline mode sẽ không ghi nhớ cho lượt sau."

		prompt_tokens = _estimate_tokens(query)
		response_tokens = _estimate_tokens(response)

		return {
			"user_id": user_id,
			"query": query,
			"route": "short_term",
			"route_reason": "baseline_no_memory",
			"route_confidence": 0.1,
			"response": response,
			"memory_hit": False,
			"token_budget": {
				"priority_1_kept": 0,
				"priority_2_kept": prompt_tokens,
				"priority_3_kept": 0,
				"priority_4_kept": 0,
				"dropped_tokens": 0,
				"budget_limit": prompt_tokens,
			},
			"token_usage": {
				"prompt_tokens": prompt_tokens,
				"response_tokens": response_tokens,
				"total_tokens": prompt_tokens + response_tokens,
			},
		}


@dataclass
class MetricAccumulator:
	turns: int = 0
	relevance_sum: float = 0.0
	relevance_count: int = 0
	matched_keywords: int = 0
	total_expected_keywords: int = 0

	context_expected_turns: int = 0
	context_used_turns: int = 0

	memory_expected_turns: int = 0
	memory_hit_turns: int = 0

	total_tokens: int = 0
	prompt_tokens: int = 0
	response_tokens: int = 0

	# Fix lỗi Pylance bằng cách dùng field factory chuẩn của Python Dataclasses
	token_budget: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _keyword_match_score(response: str, expected_keywords: List[str]) -> tuple[int, float]:
	if not expected_keywords:
		return 0, 0.0

	response_lower = response.lower()
	matched = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
	relevance = _safe_divide(matched, len(expected_keywords))
	return matched, relevance


def _format_pct(value: float) -> str:
	return f"{value * 100:.2f}%"


def _format_num(value: float) -> str:
	return f"{value:.4f}"


def _build_metrics(acc: MetricAccumulator) -> Dict[str, Any]:
	response_relevance = _safe_divide(acc.relevance_sum, acc.relevance_count)
	context_utilization = _safe_divide(acc.context_used_turns, acc.context_expected_turns)
	memory_hit_rate = _safe_divide(acc.memory_hit_turns, acc.memory_expected_turns)
	token_efficiency = _safe_divide(acc.matched_keywords, acc.total_tokens)

	return {
		"turns": acc.turns,
		"response_relevance": response_relevance,
		"context_utilization": context_utilization,
		"token_efficiency": token_efficiency,
		"memory_hit_rate": memory_hit_rate,
		"matched_keywords": acc.matched_keywords,
		"total_expected_keywords": acc.total_expected_keywords,
		"total_tokens": acc.total_tokens,
		"prompt_tokens": acc.prompt_tokens,
		"response_tokens": acc.response_tokens,
		"token_budget_breakdown": dict(acc.token_budget),
	}


def _generate_report(
	output_path: Path,
	result_payload: Dict[str, Any],
) -> None:
	with_memory = result_payload["model_results"]["with_memory"]
	without_memory = result_payload["model_results"]["without_memory"]

	def delta(metric_name: str) -> float:
		return with_memory[metric_name] - without_memory[metric_name]

	lines = [
		"# Benchmark Report - Multi-Memory Agent (Lab 17)",
		"",
		f"- Generated at (UTC): {result_payload['generated_at_utc']}",
		f"- Conversations: {result_payload['conversation_count']}",
		f"- Total turns: {result_payload['turn_count']}",
		"",
		"## KPI Comparison",
		"",
		"| Metric | With Memory | Without Memory | Delta |",
		"|---|---:|---:|---:|",
		(
			f"| Response relevance | {_format_pct(with_memory['response_relevance'])} | "
			f"{_format_pct(without_memory['response_relevance'])} | {_format_pct(delta('response_relevance'))} |"
		),
		(
			f"| Context utilization | {_format_pct(with_memory['context_utilization'])} | "
			f"{_format_pct(without_memory['context_utilization'])} | {_format_pct(delta('context_utilization'))} |"
		),
		(
			f"| Token efficiency (keyword/token) | {_format_num(with_memory['token_efficiency'])} | "
			f"{_format_num(without_memory['token_efficiency'])} | {_format_num(delta('token_efficiency'))} |"
		),
		(
			f"| Memory hit rate | {_format_pct(with_memory['memory_hit_rate'])} | "
			f"{_format_pct(without_memory['memory_hit_rate'])} | {_format_pct(delta('memory_hit_rate'))} |"
		),
		"",
		"## Token Budget Breakdown (With Memory)",
		"",
		"| Bucket | Tokens | Share |",
		"|---|---:|---:|",
	]

	breakdown = with_memory["token_budget_breakdown"]
	total_breakdown_tokens = sum(breakdown.values()) or 1
	for key in [
		"priority_1_kept",
		"priority_2_kept",
		"priority_3_kept",
		"priority_4_kept",
		"dropped_tokens",
	]:
		value = int(breakdown.get(key, 0))
		share = value / total_breakdown_tokens
		lines.append(f"| {key} | {value} | {_format_pct(share)} |")

	lines.extend(
		[
			"",
			"## Token Usage",
			"",
			f"- With memory: prompt={with_memory['prompt_tokens']}, response={with_memory['response_tokens']}, total={with_memory['total_tokens']}",
			f"- Without memory: prompt={without_memory['prompt_tokens']}, response={without_memory['response_tokens']}, total={without_memory['total_tokens']}",
		]
	)

	output_path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
	memory_agent: MultiMemoryAgent,
	dataset_path: str = "benchmark/dataset.json",
	result_output_path: str = "benchmark/results.json",
	report_output_path: str = "benchmark/Benchmark_Report.md",
) -> Dict[str, Any]:
	dataset_file = Path(dataset_path)
	dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
	conversations = dataset.get("conversations", [])

	baseline_agent = NoMemoryAgent()

	with_memory_acc = MetricAccumulator()
	without_memory_acc = MetricAccumulator()
	turn_logs: List[Dict[str, Any]] = []

	for conversation in conversations:
		conversation_id = conversation["id"]
		turns = conversation.get("turns", [])
		user_id_memory = f"memory_{conversation_id}"
		user_id_baseline = f"baseline_{conversation_id}"

		for turn_index, turn in enumerate(turns, start=1):
			query = turn["user"]
			expected_keywords: List[str] = turn.get("expected_keywords", [])
			expected_route = turn.get("expected_route", "")

			memory_result = memory_agent.chat(query=query, user_id=user_id_memory)
			baseline_result = baseline_agent.chat(query=query, user_id=user_id_baseline)

			for model_name, result, acc in [
				("with_memory", memory_result, with_memory_acc),
				("without_memory", baseline_result, without_memory_acc),
			]:
				matched_keywords, relevance = _keyword_match_score(
					response=result.get("response", ""),
					expected_keywords=expected_keywords,
				)

				token_usage = result.get("token_usage", {})
				prompt_tokens = int(token_usage.get("prompt_tokens", 0))
				response_tokens = int(token_usage.get("response_tokens", 0))
				total_tokens = int(token_usage.get("total_tokens", prompt_tokens + response_tokens))

				acc.turns += 1
				acc.total_tokens += total_tokens
				acc.prompt_tokens += prompt_tokens
				acc.response_tokens += response_tokens

				if expected_keywords:
					acc.relevance_sum += relevance
					acc.relevance_count += 1
					acc.matched_keywords += matched_keywords
					acc.total_expected_keywords += len(expected_keywords)

				if expected_route in {"long_term", "episodic", "semantic"}:
					acc.context_expected_turns += 1
					acc.memory_expected_turns += 1

					route_correct = result.get("route") == expected_route
					memory_hit = bool(result.get("memory_hit", False))

					if route_correct and memory_hit:
						acc.context_used_turns += 1
					if memory_hit:
						acc.memory_hit_turns += 1

				for budget_key, budget_value in result.get("token_budget", {}).items():
					acc.token_budget[budget_key] += int(budget_value)

				turn_logs.append(
					{
						"conversation_id": conversation_id,
						"turn_index": turn_index,
						"model": model_name,
						"query": query,
						"expected_route": expected_route,
						"expected_keywords": expected_keywords,
						"route": result.get("route"),
						"memory_hit": result.get("memory_hit", False),
						"response": result.get("response", ""),
						"matched_keywords": matched_keywords,
						"relevance": relevance,
						"token_usage": token_usage,
					}
				)

	result_payload = {
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"dataset_path": dataset_path,
		"conversation_count": len(conversations),
		"turn_count": with_memory_acc.turns,
		"model_results": {
			"with_memory": _build_metrics(with_memory_acc),
			"without_memory": _build_metrics(without_memory_acc),
		},
		"turn_logs": turn_logs,
	}

	result_file = Path(result_output_path)
	result_file.parent.mkdir(parents=True, exist_ok=True)
	result_file.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

	report_file = Path(report_output_path)
	report_file.parent.mkdir(parents=True, exist_ok=True)
	_generate_report(output_path=report_file, result_payload=result_payload)

	return result_payload