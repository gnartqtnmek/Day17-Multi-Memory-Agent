from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from agent.nodes import AgentNodes
from agent.state import AgentState


def build_graph(nodes: AgentNodes):
	workflow = StateGraph(AgentState)

	workflow.add_node("route", nodes.route_query)
	workflow.add_node("retrieve", nodes.retrieve_context)
	workflow.add_node("respond", nodes.generate_response)
	workflow.add_node("persist", nodes.persist_memory)

	workflow.add_edge(START, "route")
	workflow.add_edge("route", "retrieve")
	workflow.add_edge("retrieve", "respond")
	workflow.add_edge("respond", "persist")
	workflow.add_edge("persist", END)

	return workflow.compile()


class MultiMemoryAgent:
	def __init__(self, nodes: AgentNodes) -> None:
		self._nodes = nodes
		self._graph = build_graph(nodes)

	def chat(self, query: str, user_id: str = "default") -> Dict[str, Any]:
		initial_state: AgentState = {
			"user_id": user_id,
			"query": query,
		}
		result = self._graph.invoke(initial_state)
		return dict(result)

