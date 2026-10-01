"""Iterative (looping) graph demo built with LangGraph + OpenRouter.

Exposes:
    build_graph()  -> the compiled iterative StateGraph
    run(topic)     -> run the graph and print an iteration-by-iteration trace
    main()         -> console-script entry point
"""

from .iterative_graph import build_graph, run, main

__all__ = ["build_graph", "run", "main"]
