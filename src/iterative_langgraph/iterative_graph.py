"""
================================================================================
 ITERATIVE GRAPHS WITH LANGGRAPH  +  OPENROUTER
================================================================================

WHAT IS AN "ITERATIVE GRAPH"?
-----------------------------
A normal LangChain chain is a straight line (a DAG):

        input --> prompt --> llm --> output        (runs exactly once)

An *iterative* (a.k.a. cyclic / looping) graph lets control flow go BACKWARDS:

        input --> generate --> critique --+
                     ^                    |
                     |   (not good yet)  |
                     +------ revise <----+

The graph keeps looping over a "critique -> revise" cycle until some stop
condition is met (the output is good enough, OR we hit a max-iteration cap).

LangGraph is the part of the LangChain ecosystem purpose-built for this: you
model your app as a state machine of NODES (python functions) and EDGES
(the wiring between them), and LangGraph handles the looping, the shared
state, and the safety limits for you.

THE REFLECTION / SELF-REFINEMENT PATTERN USED HERE
--------------------------------------------------
  1. generate  -> write a first draft from a topic
  2. critique  -> an LLM "critic" reviews the draft and returns a verdict
  3. revise    -> an LLM rewrites the draft using the critic's feedback
  4. router    -> loop back to step 2, unless approved OR max rounds reached

This is the same idea behind "Reflection" agents, self-refine, CRITIC, etc.

--------------------------------------------------------------------------------
RUN IT
--------------------------------------------------------------------------------
  uv run python -m iterative_langgraph
  (or)  uv run iterative-langgraph

Config comes from the .env file in the project root:
  OPENROUTER_API_KEY=sk-or-v1-...        # required
  OPENROUTER_MODEL=<any-or-slug>:free    # optional, defaults to a FREE model
  MAX_ITERATIONS=3                       # optional, loop safety cap

This uses the DEDICATED OpenRouter integration (`langchain_openrouter.
ChatOpenRouter`) -- NOT the generic OpenAI client -- with a FREE model.
================================================================================
"""

from __future__ import annotations

# `operator.add` is a REDUCER we attach to a state field (explained below).
import operator

# `os` reads environment variables (the API key + model name).
import os

# `TypedDict` gives our graph state a concrete, typed schema.
# `Annotated` lets us attach metadata (the reducer) to a single state field.
from typing import Annotated, TypedDict

# `load_dotenv()` reads the project's .env file into os.environ.
from dotenv import load_dotenv

# LangChain message objects. `SystemMessage` = instructions/role,
# `HumanMessage` = the user's turn (here, the data we hand the model).
from langchain_core.messages import HumanMessage, SystemMessage

# `ChatOpenRouter` is the DEDICATED LangChain integration for OpenRouter
# (from the official `langchain-openrouter` package). Unlike the generic
# ChatOpenAI client, it is purpose-built for OpenRouter: it already points at
# OpenRouter's endpoint and exposes OpenRouter-native features like app
# attribution, provider routing, and plugins.
from langchain_openrouter import ChatOpenRouter

# The 4 building blocks of every LangGraph app:
#   StateGraph   -> the graph builder class
#   START / END  -> special sentinel nodes (the entry / exit points)
from langgraph.graph import END, START, StateGraph


# ==============================================================================
# 0. CONFIG  --  load secrets + tunables from the environment / .env
# ==============================================================================

load_dotenv()  # populate os.environ from ./.env

# NOTE: ChatOpenRouter already defaults its endpoint to OpenRouter
# (https://openrouter.ai/api/v1), so we do NOT hard-code a base_url.

# A FREE OpenRouter model: a slug ending in ":free" costs $0 in / $0 out.
# Browse the free models here: https://openrouter.ai/models?max_price=0
# Override with the OPENROUTER_MODEL env var to use any other OpenRouter slug.
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

# The hard cap on how many critique->revise rounds we allow.
# THIS IS THE KEY SAFETY VALVE: without a cap an imperfect critic could make
# the graph loop forever (and burn through your credits).
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))

# The thing we ask the model to write about. Change it to try other topics.
TOPIC = os.getenv(
    "TOPIC",
    "Why iterative (looping) graphs matter for building reliable LLM agents",
)

# ==============================================================================
# 1. THE MODEL  --  one function that returns a configured OpenRouter client
# ==============================================================================

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


# A single, module-level client reused by every node.
llm = build_llm()


# ==============================================================================
# 2. THE STATE  --  the single shared "memory" every node reads from / writes
# ==============================================================================
#
# In LangGraph, state is a typed dict. Each NODE receives the current state and
# returns a PARTIAL update (only the keys it changed). LangGraph merges that
# update into the running state automatically.
#
# By default a field is OVERWRITTEN by the newest value. You can attach a
# REDUCER with `Annotated[..., reducer]` to change HOW updates merge. We use
# `operator.add` on `history` so each node APPENDS to the log instead of
# replacing it -- giving us a full, ordered trail of every iteration.

class RefineState(TypedDict):
    topic: str          # the subject we're writing about (constant input)
    draft: str          # the current best version of the text
    critique: str       # the latest critic feedback
    iteration: int      # how many drafts we've produced so far
    approved: bool      # did the critic sign off on the current draft?
    history: Annotated[list[str], operator.add]  # append-only audit log


# ==============================================================================
# 3. THE NODES  --  plain python functions: (state) -> partial-update dict
# ==============================================================================

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


# A strict instruction that forces the critic to end with a machine-readable
# verdict line. We parse that line to decide whether to loop or stop.
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


# ==============================================================================
# 4. THE ROUTER  --  the conditional edge that decides loop vs. stop
# ==============================================================================
#
# A CONDITIONAL EDGE calls this function and uses its RETURN STRING as the name
# of the next node. That's how a graph "branches", and by routing back to a
# previous node we create the CYCLE that makes this graph iterative.

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


# ==============================================================================
# 5. BUILD THE GRAPH  --  register nodes, then wire the edges
# ==============================================================================

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


# ==============================================================================
# 6. RUN IT  --  execute the loop and print a human-readable trace
# ==============================================================================

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
    # Initial state: all keys present. `history` starts empty and grows via
    # its `operator.add` reducer as nodes run.
    initial_state: RefineState = {
        "topic": topic,
        "draft": "",
        "critique": "",
        "iteration": 0,
        "approved": False,
        "history": [],
    }

    config = {"recursion_limit": 50}

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


# Allows `python src/iterative_langgraph/iterative_graph.py` too.
if __name__ == "__main__":
    main()
