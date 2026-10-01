# Iterative Graph with LangGraph + OpenRouter

A minimal, heavily-commented example of an **iterative (looping / cyclic) graph**
built with [LangGraph](https://langchain-ai.github.io/langgraph/) and an LLM served
through [OpenRouter](https://openrouter.ai/).

## The idea

Most LLM "chains" are straight lines that run exactly once. An **iterative graph**
lets control flow loop back so the model can refine its own output:

```
START --> generate --> critique --(not good enough)--> revise --+
                        ^                                       |
                        |                                       |
                        +---------------------------------------+
                        |
                 (approved OR max rounds) --> END
```

`critique -> revise -> critique -> revise -> ...` repeats until either the
critic approves the draft **or** a hard iteration cap (`MAX_ITERATIONS`) is hit.
This is the mechanism behind "reflection" / self-refinement agents.

## Concepts demonstrated

| Concept | Where |
|---|---|
| `StateGraph` + typed state | `RefineState` |
| Append-only field via **reducer** (`Annotated[..., operator.add]`) | `history` |
| Nodes as `(state) -> partial_update` functions | `generate_node`, `critique_node`, `revise_node` |
| **Conditional edge** / router (the loop decision) | `route_after_critique` |
| The **cycle** (edge back to a previous node) | `revise -> critique` |
| Runaway-loop safety | `MAX_ITERATIONS` + `config["recursion_limit"]` |
| Streaming a live trace | `graph.stream(..., stream_mode="values")` |
| Dedicated OpenRouter integration (`ChatOpenRouter`) + FREE model | `build_llm()` |

## Why `ChatOpenRouter` (not `ChatOpenAI`)?

This uses the **dedicated** `langchain-openrouter` integration, so the model
object is genuinely OpenRouter-aware (native app attribution, provider routing,
plugins) instead of a generic OpenAI client pointed at a different URL:

```python
from langchain_openrouter import ChatOpenRouter

llm = ChatOpenRouter(
    model="nvidia/nemotron-3-super-120b-a12b:free",  # a FREE model
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

### Using free models

On OpenRouter, a model slug ending in **`:free`** costs `$0` for both input and
output tokens. Browse them at
<https://openrouter.ai/models?max_price=0>, then set `OPENROUTER_MODEL` (or the
default in `iterative_graph.py`) to any slug you like. Free models can be a bit
slower and are subject to rate limits, but they need no credits.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies are declared in
`pyproject.toml`; install/sync them with:

```bash
uv sync
```

Create a `.env` file in the project root:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-...
# optional overrides
# OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
# MAX_ITERATIONS=3
# TOPIC=Why iterative graphs matter
```

## Run

```bash
uv run python -m iterative_langgraph
# or
uv run iterative-langgraph
```

You'll get a step-by-step trace of each loop iteration plus the final draft.

## Files

```
src/iterative_langgraph/
  iterative_graph.py   # the full, commented example (start here)
  __init__.py          # exports build_graph / run / main
  __main__.py          # enables `python -m iterative_langgraph`
```
