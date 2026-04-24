from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from agent import AgentNodes, MemoryRouter, MultiMemoryAgent
from benchmark import run_benchmark
from memory import EpisodicMemory, LongTermMemory, SemanticMemory, ShortTermMemory
from dotenv import load_dotenv

load_dotenv()


def build_memory_agent(token_limit: int = 900) -> MultiMemoryAgent:
	short_term = ShortTermMemory()
	long_term = LongTermMemory(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
	episodic = EpisodicMemory(file_path=os.getenv("EPISODIC_FILE", "data/episodic_log.json"))
	semantic = SemanticMemory(
		persist_directory=os.getenv("CHROMA_DIR", "data/chroma"),
		collection_name=os.getenv("CHROMA_COLLECTION", "semantic_knowledge"),
	)
	semantic.seed_default_knowledge()

	nodes = AgentNodes(
		short_term=short_term,
		long_term=long_term,
		episodic=episodic,
		semantic=semantic,
		router=MemoryRouter(),
		token_limit=token_limit,
	)
	return MultiMemoryAgent(nodes=nodes)


def run_interactive_chat(agent: MultiMemoryAgent, user_id: str) -> None:
	print("Multi-Memory Agent (LangGraph) - type 'exit' to stop")

	while True:
		query = input("You: ").strip()
		if not query:
			continue
		if query.lower() in {"exit", "quit"}:
			print("Session ended.")
			break

		result = agent.chat(query=query, user_id=user_id)
		print(f"Agent: {result.get('response', '')}")
		print(
			"Debug: "
			f"route={result.get('route')} | "
			f"memory_hit={result.get('memory_hit')} | "
			f"tokens={result.get('token_usage', {}).get('total_tokens', 0)}"
		)


def run_single_turn(agent: MultiMemoryAgent, user_id: str, query: str) -> None:
	result: Dict[str, Any] = agent.chat(query=query, user_id=user_id)
	printable = {
		"response": result.get("response"),
		"route": result.get("route"),
		"route_reason": result.get("route_reason"),
		"memory_hit": result.get("memory_hit"),
		"token_usage": result.get("token_usage"),
		"token_budget": result.get("token_budget"),
	}
	print(json.dumps(printable, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Multi-Memory Agent Lab 17")
	subparsers = parser.add_subparsers(dest="command")

	chat_parser = subparsers.add_parser("chat", help="Run interactive or single-turn chat")
	chat_parser.add_argument("--user-id", default="demo_user")
	chat_parser.add_argument("--query", default="")
	chat_parser.add_argument("--token-limit", type=int, default=900)

	benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark report")
	benchmark_parser.add_argument("--dataset", default="benchmark/dataset.json")
	benchmark_parser.add_argument("--results", default="benchmark/results.json")
	benchmark_parser.add_argument("--report", default="benchmark/Benchmark_Report.md")
	benchmark_parser.add_argument("--token-limit", type=int, default=900)

	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.command == "benchmark":
		agent = build_memory_agent(token_limit=args.token_limit)
		summary = run_benchmark(
			memory_agent=agent,
			dataset_path=args.dataset,
			result_output_path=args.results,
			report_output_path=args.report,
		)
		print(json.dumps(summary["model_results"], ensure_ascii=False, indent=2))
		return

	user_id = getattr(args, "user_id", "demo_user")
	query = getattr(args, "query", "")
	token_limit = getattr(args, "token_limit", 900)

	agent = build_memory_agent(token_limit=token_limit)
	if query:
		run_single_turn(agent=agent, user_id=user_id, query=query)
	else:
		run_interactive_chat(agent=agent, user_id=user_id)


if __name__ == "__main__":
	main()

