from __future__ import annotations
import operator
import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph

load_dotenv()
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
TOPIC = os.getenv(
    "TOPIC",
    "Why iterative (looping) graphs matter for building reliable LLM agents",
)

def build_llm(temperature: float = 0.7) -> ChatOpenRouter:
    """
    Create a LangChain chat model using the DEDICATED OpenRouter integration.

    `ChatOpenRouter` (from `langchain-openrouter`) is purpose-built for
    OpenRouter, so there is no base_url juggling -- it already targets
    https://openrouter.ai/api/v1. We just pass:
      * our OpenRouter key as `api_key`, and
      * the model slug (here a FREE one, e.g. "...:free").
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        # Fail loudly and early with a helpful message instead of a cryptic 401.
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file, e.g.\n"
            "  OPENROUTER_API_KEY=sk-or-v1-...\n"
            "Get one at https://openrouter.ai/keys"
        )

    return ChatOpenRouter(
        model=MODEL,          # alias for `model_name`
        api_key=api_key, #type:ignore      # alias for `openrouter_api_key`
        temperature=temperature,
        max_retries=2,        # retry transient network / 429 errors
        # Optional OpenRouter attribution (shows up on their leaderboard).
        app_title="Iterative LangGraph Demo",
        app_url="https://github.com/local/iterative-langgraph",
    )

llm = build_llm()

class RefineState(TypedDict):
    topic: str          # the subject we're writing about (constant input)
    draft: str          # the current best version of the text
    critique: str       # the latest critic feedback
    iteration: int      # how many drafts we've produced so far
    approved: bool      # did the critic sign off on the current draft?
    history: Annotated[list[str], operator.add]  # append-only audit log

def generate_node(state: RefineState) -> dict:
    """
    NODE 1: produce the first draft from the topic.

    Seeding `iteration=1` and logging to `history` starts the loop counter.
    Because `history` uses `operator.add`, the returned list is APPENDED to
    whatever is already in state (here it's empty, so we just get one item).
    """
    messages = [
        SystemMessage(
            "You are a concise, expert technical writer. "
            "Write clear prose with short paragraphs. No markdown headings."
        ),
        HumanMessage(content=f"Write a first draft on this topic:\n\n{state['topic']}"),
    ]
    draft = llm.invoke(messages).content

    # Return ONLY the keys this node changed; LangGraph merges the rest.
    return {
        "draft": draft,
        "iteration": 1,
        "approved": False,
        "history": [f"[generate] initial draft written ({len(draft)} chars)"],
    }

CRITIC_SYSTEM_PROMPT = """\
You are a demanding but fair editor. Review the draft you are given and list
the most important, concrete improvements it still needs.

Rules:
- Be specific: quote the weak part, then say how to fix it.
- If the draft is genuinely strong and needs no meaningful changes, say so.
- You MUST end your reply with exactly one final line, either:
    VERDICT: APPROVED
  or
    VERDICT: REVISE
"""


def critique_node(state: RefineState) -> dict:
    """
    NODE 2: review the current draft and decide "good enough?".

    We parse the required trailing "VERDICT:" line into a clean boolean so the
    ROUTER (step 4) can make a simple, deterministic decision. Keeping the
    decision logic OUT of the LLM's prose is what makes the loop reliable.
    """
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=f"Here is the draft to review:\n\n{state['draft']}"),
    ]
    critique = llm.invoke(messages).content

    # Look for the verdict marker anywhere in the reply (case-insensitive).
    approved = "VERDICT: APPROVED" in critique.upper() #type:ignore

    return {
        "critique": critique,
        "approved": approved,
        "history": [f"[critique] iteration={state['iteration']} approved={approved}"],
    }


def revise_node(state: RefineState) -> dict:
    """
    NODE 3: rewrite the draft, applying the critic's feedback.

    This is the node that closes the loop. Note how we pass BOTH the current
    draft and the critique -- the model EDITS rather than starts over.
    We also bump `iteration`, which the router uses as the hard stop.
    """
    messages = [
        SystemMessage(
            "You are an expert editor. Rewrite the draft so that it fully "
            "addresses every point in the critique. Keep what already works, "
            "fix the rest. Output ONLY the improved draft, nothing else."
        ),
        HumanMessage(
            content=(
                f"Current draft:\n\n{state['draft']}\n\n"
                f"Editor's critique to address:\n\n{state['critique']}"
            )
        ),
    ]
    revised = llm.invoke(messages).content
    next_iteration = state["iteration"] + 1

    return {
        "draft": revised,
        "iteration": next_iteration,
        "history": [f"[revise] produced draft v{next_iteration}"],
    }

def route_after_critique(state: RefineState) -> str:
    """
    Decide where to go after a critique.

    Returns:
        "revise" -> loop back for another round
        "end"    -> stop the graph (we're done)
    """
    # Stop condition A: the critic approved the draft.
    if state["approved"]:
        return "end"

    # Stop condition B: we've hit the safety cap -- stop even if not approved,
    # otherwise an unsatisfiable critic would loop forever.
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"

    # Otherwise: iterate. Go revise, which loops back into critique.
    return "revise"

def build_graph():
    """
    Construct, wire and COMPILE the iterative graph.

    compile() validates the topology and returns a runnable object.
    """
    # Create a graph builder bound to our state schema.
    builder = StateGraph(RefineState)

    # --- Register the nodes (name -> function) --------------------------------
    builder.add_node("generate", generate_node)
    builder.add_node("critique", critique_node)
    builder.add_node("revise", revise_node)

    # --- Wire the edges -------------------------------------------------------

    # Entry point: START -> generate (runs once at the beginning).
    builder.add_edge(START, "generate")

    # Then always go generate -> critique.
    builder.add_edge("generate", "critique")

    # After critique, use the ROUTER to choose the next node. The third arg maps
    # the router's return value -> the real next node:
    #   "revise" -> the revise node
    #   "end"    -> the END sentinel (finishes the run)
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {"revise": "revise", "end": END},
    )

    # THE CYCLE: after revising, go back to critique.
    # critique -> revise -> critique -> revise -> ... until the router says END.
    builder.add_edge("revise", "critique")

    # Compile into a runnable graph.
    return builder.compile()


graph = build_graph()

def run(topic: str = TOPIC) -> RefineState:
    """
    Run the graph ONCE and print an iteration-by-iteration trace.

    We use `stream_mode="values"`, which yields the FULL state snapshot after
    every step. That gives us two things in a single pass:
      * a live trace (print whatever changed since the previous snapshot), and
      * the final state for free (the last snapshot) -- so we never have to run
        the graph a second time with `.invoke()`.

    IMPORTANT: LangGraph guards against runaway loops with a `recursion_limit`
    in the config (default 25 super-steps). Our MAX_ITERATIONS cap keeps us
    well under that, but you can raise `recursion_limit` for longer loops.
    """
    
    initial_state: RefineState = {
        "topic": topic,
        "draft": "",
        "critique": "",
        "iteration": 0,
        "approved": False,
        "history": [],
    }

    config = {"recursion_limit": 10}

    print("=" * 80)
    print(f"TOPIC: {topic}")
    print(f"MODEL: {MODEL}   |   MAX_ITERATIONS: {MAX_ITERATIONS}")
    print("=" * 80)

    final_state: RefineState = initial_state
    seen_history = 0          # how many history lines we've already printed
    last_draft = ""           # last draft text we printed a preview for

    # Each `snapshot` is the complete state after one node ran.
    for snapshot in graph.stream(initial_state, config=config, stream_mode="values"): #type:ignore
        final_state = snapshot  # keep the latest; last one == final state #type:ignore

        # 1) Print any log lines appended since the previous step.
        new_lines = snapshot["history"][seen_history:]
        seen_history = len(snapshot["history"])
        for line in new_lines:
            print(f"  {line}")

        # 2) Print a draft preview whenever the draft text changed.
        if snapshot["draft"] and snapshot["draft"] != last_draft:
            last_draft = snapshot["draft"]
            preview = last_draft.replace("\n", " ").strip()
            print(f"  >>> draft: {preview[:220]}...")

    print("-" * 80)
    print(
        f"FINISHED after {final_state['iteration']} draft(s). "
        f"approved={final_state['approved']}"
    )
    print(f"total nodes executed: {len(final_state['history'])}")
    print("-" * 80)
    print("FINAL DRAFT:")
    print(final_state["draft"])
    print("=" * 80)
    return final_state


def main() -> None:
    """Console-script entry point (see [project.scripts] in pyproject.toml)."""
    run()

if __name__ == "__main__":
    main()
