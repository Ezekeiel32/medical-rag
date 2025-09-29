#!/usr/bin/env python3
"""
MLE-STAR Multi-Agent Orchestrator (Google ADK)

This script builds a minimal, real MLE-STAR-style multi-agent workflow using
Google ADK LlmAgent + FunctionTool and runs one planning/refinement/evaluation
loop against the Hebrew OCR RAG system.

- Planner/Search agent: plans steps and can call indexing/dataset tools
- Refiner agent: selects next embedding model and re-indexes
- Evaluator agent: builds RAGAS dataset and evaluates
- Coordinator agent: orchestrates sub-agents via AgentTool

Outputs are printed to stdout. Session state carries `ocr_dir` and
`current_model` across tools/agents.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types


# Ensure project is importable
PROJECT_ROOT = "/home/chezy/rag_medical"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---- Tool wrappers that read defaults from session state ----
def _state(tool_context) -> dict[str, Any]:
    try:
        return tool_context.session_context.state or {}
    except Exception:
        return {}


def tool_index(
    ocr_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    chunk_chars: int = 900,
    overlap_chars: int = 120,
    enable_multi_vector: bool = True,
    tool_context=None,
) -> dict[str, Any]:
    """Index OCR corpus with the specified embedding model.

    Falls back to session state for `ocr_dir` and `model_name` if missing.
    """
    from src.rag_ocr import index_ocr_corpus

    st = _state(tool_context)
    # Prefer session state; only accept absolute incoming path
    ocr_dir = (ocr_dir if (ocr_dir and os.path.isabs(ocr_dir)) else st.get("ocr_dir")) or "/home/chezy/rag_medical/ocr_out"
    model_name = model_name or st.get("current_model") or "intfloat/multilingual-e5-large"
    index_path, meta_path = index_ocr_corpus(
        ocr_dir,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
        model_name=model_name,
        enable_multi_vector=enable_multi_vector,
    )
    # Persist useful state
    try:
        st["current_model"] = model_name
        st["last_index"] = index_path
        st["last_meta"] = meta_path
        tool_context.session_context.state = st
    except Exception:
        pass
    return {"index": index_path, "meta": meta_path, "model": model_name}


def tool_build_ragas(
    ocr_dir: Optional[str] = None,
    model: str = "qwen2.5:7b-instruct",
    top_k: int = 6,
    tool_context=None,
) -> str:
    """Build Hebrew RAGAS dataset at ocr_dir/ragas_he.jsonl."""
    from src.ocr_ragas_eval import build_hebrew_ragas_samples, save_jsonl

    st = _state(tool_context)
    ocr_dir = (ocr_dir if (ocr_dir and os.path.isabs(ocr_dir)) else st.get("ocr_dir")) or "/home/chezy/rag_medical/ocr_out"
    # Ensure index exists; if missing, build it using current model
    embed_dir = os.path.join(ocr_dir, "ocr_embeddings")
    index_path = os.path.join(embed_dir, "index.faiss")
    if not os.path.exists(index_path):
        from src.rag_ocr import index_ocr_corpus
        current_model = st.get("current_model") or "intfloat/multilingual-e5-large"
        index_ocr_corpus(ocr_dir, chunk_chars=900, overlap_chars=120, model_name=current_model, enable_multi_vector=True)
    samples = build_hebrew_ragas_samples(ocr_dir, model=model, top_k=top_k)
    out_path = os.path.join(ocr_dir, "ragas_he.jsonl")
    save_jsonl(samples, out_path)
    return out_path


def tool_run_ragas(
    jsonl_path: Optional[str] = None,
    ocr_dir: Optional[str] = None,
    tool_context=None,
) -> dict[str, float]:
    """Run RAGAS over the given jsonl (defaults to ocr_dir/ragas_he.jsonl)."""
    from src.ragas_eval import run_ragas

    st = _state(tool_context)
    ocr_dir = (ocr_dir if (ocr_dir and os.path.isabs(ocr_dir)) else st.get("ocr_dir")) or "/home/chezy/rag_medical/ocr_out"
    default_jsonl = os.path.join(ocr_dir, "ragas_he.jsonl")
    # Only accept absolute provided path; otherwise use default
    jsonl_path = jsonl_path if (jsonl_path and os.path.isabs(jsonl_path)) else default_jsonl
    # If dataset missing, build it first
    if not os.path.exists(jsonl_path):
        from src.ocr_ragas_eval import build_hebrew_ragas_samples, save_jsonl
        samples = build_hebrew_ragas_samples(ocr_dir, model="qwen2.5:7b-instruct", top_k=6)
        save_jsonl(samples, jsonl_path)
    return run_ragas(jsonl_path)


def build_agents() -> Runner:
    """Construct the ADK multi-agent graph and return a Runner."""
    # Wrap functions as ADK tools
    index_tool = FunctionTool(tool_index)
    # Keep names equal to function names so LLM calls match
    index_tool.name = "tool_index"

    build_ragas_tool = FunctionTool(tool_build_ragas)
    build_ragas_tool.name = "tool_build_ragas"

    run_ragas_tool = FunctionTool(tool_run_ragas)
    run_ragas_tool.name = "tool_run_ragas"

    # Specialized agents
    search_planner = LlmAgent(
        name="SearchPlanner",
        model="gemini-2.0-flash",
        instruction=(
            "Plan MLE-STAR steps for Hebrew OCR RAG."
            " Use tools to index, build dataset, and evaluate."
            " Keep outputs concise."
        ),
        tools=[index_tool, build_ragas_tool, run_ragas_tool],
    )

    refiner = LlmAgent(
        name="Refiner",
        model="gemini-2.0-flash",
        instruction=(
            "Given current metrics, propose the next embedding model to try"
            " (e.g., mpnet/LaBSE/e5-base) and call index_ocr_corpus."
        ),
        tools=[index_tool],
    )

    evaluator = LlmAgent(
        name="Evaluator",
        model="gemini-2.0-flash",
        instruction=(
            "Use build_ragas_dataset then run_ragas_eval to compute faithfulness"
            " and answer_relevancy; summarize briefly."
        ),
        tools=[build_ragas_tool, run_ragas_tool],
    )

    # Coordinator can call the sub-agents as tools
    coordinator = LlmAgent(
        name="MLE_STAR_Coordinator",
        model="gemini-2.0-flash",
        instruction=(
            "You orchestrate MLE-STAR for Hebrew OCR RAG: plan -> refine ->"
            " evaluate. Run one full loop and present the best next step."
        ),
        tools=[AgentTool(agent=search_planner), AgentTool(agent=refiner), AgentTool(agent=evaluator)],
    )

    runner = Runner(
        app_name="rag_mle_star",
        agent=coordinator,
        session_service=InMemorySessionService(),
    )
    return runner


def main() -> int:
    runner = build_agents()

    # Session state seeds
    state = {
        "ocr_dir": "/home/chezy/rag_medical/ocr_out",
        "current_model": "intfloat/multilingual-e5-large",
    }

    # Create session
    sess = runner.session_service.create_session_sync(
        app_name="rag_mle_star", user_id="user", state=state
    )

    # Kick off with a specific instruction for one MLE-STAR loop
    # Build genai Content/Part manually for compatibility
    user_msg = types.Content(
        role="user",
        parts=[types.Part(text=(
            "Run an MLE-STAR loop: 1) index with current_model,"
            " 2) build RAGAS dataset with qwen2.5:7b-instruct,"
            " 3) evaluate ragas_he.jsonl, 4) suggest next embedding to try."
        ))],
    )

    # Stream events and print final LLM responses
    for ev in runner.run(user_id="user", session_id=sess.id, new_message=user_msg):
        try:
            et = getattr(ev, "type", "")
            if str(et).endswith("llm_response"):
                cand = ev.response.candidates[0]
                txt = "".join(getattr(p, "text", "") for p in cand.content.parts)
                print(json.dumps({"event": str(et), "text": txt}, ensure_ascii=False))
        except Exception:
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


