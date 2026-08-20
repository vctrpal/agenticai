import os
import argparse
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

load_dotenv()


class ResearchState(TypedDict):
    query: str
    results: list[str]
    messages: Annotated[list, add_messages]
    report:str

def search_web(state:ResearchState)->ResearchState:
    # Implement web search logic here
    tavily_api_key =  os.getenv("TAVILY_API_KEY")

    print(f"Tavily API Key: {tavily_api_key}")
    tools = TavilySearch(max_results=5,tavily_api_key="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")
    raw_results = tools.invoke(state["query"])
    print(f"Raw search results: {raw_results}")
    if isinstance(raw_results, list):
        state["results"] = raw_results
    elseif  isinstance(raw_results, dict) and "results" in raw_results:
        state["results"] = raw_results.get("results", [])   
    else:
        state["results"] = []

    return state;    

def synthesize_report(state:ResearchState)->ResearchState:
     llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)
    graph.add_node("search", search_web)
   ## graph.add_node("synthesize", synthesize_report)
    graph.set_entry_point("search")
  ##  graph.add_edge("search", "synthesize")
    graph.add_edge("search", END)
    return graph.compile()

def main():
    parser = argparse.ArgumentParser(description="Web Research Agent")
    parser.add_argument("--query", default="latest advances in AI agents 2024", help="Research query")
    args = parser.parse_args()
    print(f"Running research agent with query: {args.query}")

    agent = build_graph()
    result = agent.invoke({"query": args.query, "results": [], "messages": [], "report": ""})



if __name__ == "__main__":
    main()

