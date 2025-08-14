# bfcl_slime/rollout_bfcl.py
import os, json, random, time, pathlib, logging, itertools, re
from typing import List, Dict, Any
import requests

# SLIME expects this signature. It will import this symbol via --rollout-function-path
# Return: list[list[SampleDict]] where each inner list is a GRPO group
def generate_rollout(args, rollout_id, buffer=None, *, evaluation: bool = False):
    """
    Required args used here (pass via SLIME CLI):
      --prompt-data: path to a jsonl that *indexes* BFCL items or leave unset to sample from BFCL package
      --rollout-batch-size: number of BFCL items per rollout (N prompts)
      --n-samples-per-prompt: GRPO group size (G)
      --rollout-max-response-len, --rollout-temperature, etc. are respected

    Extra env vars / args this file uses:
      BFCL_CATEGORY           (e.g., 'multi_turn_base', 'multi_turn_miss_param', 'multi_turn_miss_function')
      BFCL_PROJECT_ROOT       (optional; if unset we use bfcl-eval package data)
      BFCL_HANDLER_NAME       ('openai' default; see bfcl-eval SUPPORTED_MODELS.md)
      SGLANG_ROUTER_URL       (e.g., 'http://127.0.0.1:30000/v1/chat/completions')
      BFCL_MAX_TURNS          (default 3)
      BFCL_MAX_STEPS_PER_TURN (default 6)
    """
    log = logging.getLogger("bfcl_rollout")
    log.setLevel(logging.INFO)

    # Router & sampling params
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    router_url = f"{url}/v1/chat/completions"
    temperature = getattr(args, "rollout_temperature", 0.8)
    top_p = getattr(args, "rollout_top_p", 1.0)
    max_new_tokens = getattr(args, "rollout_max_response_len", 1024)
    group_size = getattr(args, "n_samples_per_prompt", 8)
    batch_size = getattr(args, "rollout_batch_size", 32)

    # BFCL knobs
    category = os.getenv("BFCL_CATEGORY", "multi_turn_base")
    max_turns = int(os.getenv("BFCL_MAX_TURNS", "3"))
    max_steps_per_turn = int(os.getenv("BFCL_MAX_STEPS_PER_TURN", "6"))
    handler_name = os.getenv("BFCL_HANDLER_NAME", "openai")  # used in reward stage

    # Load BFCL data entries (id, question, func_doc)
    bfcl_items = _load_bfcl_multi_turn_items(category)
    selected = random.sample(bfcl_items, k=min(batch_size, len(bfcl_items)))

    # Build GRPO groups
    grpo_groups: List[List[Dict[str, Any]]] = []
    for item in selected:
        # Build G samples for the same prompt (GRPO group)
        group_samples: List[Dict[str, Any]] = []
        for g in range(group_size):
            convo_log_raw = _simulate_multi_turn(
                router_url=router_url,
                question=item["question"],
                func_doc=item["func_doc"],
                max_turns=max_turns,
                max_steps_per_turn=max_steps_per_turn,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            # BFCL wants: result = [[step_text, step_text, ...], [ ... turn2 ... ], ...]
            bfcl_result_payload = {
                "id": item["id"],
                "result": convo_log_raw,   # list-of-lists of raw assistant outputs
                # attach hints for the reward function
                "bfcl_meta": {
                    "category": category,
                    "handler_name": handler_name,
                    "question": item["question"],
                    "func_doc": item["func_doc"],
                },
            }

            # SLIME Sample dict minimal fields (tokens/response_length may be filled by backend if logprobs returned;
            # here we keep placeholders. If your SGLang returns token ids/logprobs, attach them in 'tokens'.)
            sample = {
                "prompt": item["question"],             # optional: full serialized messages if you prefer
                "metadata": bfcl_result_payload,        # we’ll read it in reward stage
                "tokens": None,                         # let SLIME handle with SGLang logprobs if configured
                "response_length": sum(len(" ".join(t)) for t in convo_log_raw),
                "reward": 0.0,                          # filled by reward model later
                "truncated": False,
            }
            group_samples.append(sample)
        grpo_groups.append(group_samples)
    return grpo_groups


# -----------------------------
# Helpers
# -----------------------------
def _load_bfcl_multi_turn_items(category: str) -> List[Dict[str, Any]]:
    """
    Loads BFCL v3 multi-turn items from the bfcl-eval package data (preferred).
    Falls back to BFCL_PROJECT_ROOT clone if provided.

    Returns list of dicts: {id, question, func_doc}
    """
    # Try package data first
    try:
        import importlib.resources as ir
        pkg = "bfcl_eval"  # pip package ships evaluator and data
        data_dir = ir.files(pkg) / "data"
        # dataset file names match BFCL_v3_<category>.json or .jsonl
        for ext in (".json", ".jsonl"):
            p = data_dir / f"BFCL_v3_{category}{ext}"
            if p.exists():
                return _read_bfcl_file(p)
    except Exception as e:
        import logging
        logging.getLogger("bfcl_rollout").warning(f"Failed to load from package: {e}")

    # Fallback to a local clone via env var
    project_root = os.getenv("BFCL_PROJECT_ROOT")
    if not project_root:
        raise FileNotFoundError(
            "Could not locate BFCL dataset inside bfcl-eval package. "
            "Set BFCL_PROJECT_ROOT to your gorilla/berkeley-function-call-leaderboard path."
        )
    
    # Try multiple possible paths
    possible_paths = [
        pathlib.Path(project_root) / "berkeley-function-call-leaderboard" / "data" / f"BFCL_v3_{category}.json",
        pathlib.Path(project_root) / f"BFCL_v3_{category}.json",
        pathlib.Path(project_root) / "bfcl" / f"BFCL_v3_{category}.json",
    ]
    
    for p in possible_paths:
        if p.exists():
            return _read_bfcl_file(p)
    
    # If we're here, file not found - create a simple mock dataset for testing
    import logging
    log = logging.getLogger("bfcl_rollout")
    log.warning(f"BFCL dataset not found for category '{category}'. Creating mock data for testing.")
    
    # Create minimal mock data
    mock_items = []
    for i in range(10):  # Create 10 mock items
        mock_items.append({
            "id": f"mock_{category}_{i}",
            "question": f"Test question {i}: Calculate the sum of 5 + 3",
            "func_doc": json.dumps([{
                "name": "calculate_sum",
                "description": "Calculate the sum of two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            }])
        })
    return mock_items


def _read_bfcl_file(path: pathlib.Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                rec = json.loads(line)
                items.append({"id": rec["id"], "question": rec["question"], "func_doc": rec["func_doc"]})
        else:
            data = json.load(f)
            for rec in data:
                items.append({"id": rec["id"], "question": rec["question"], "func_doc": rec["func_doc"]})
    return items


def _simulate_multi_turn(
    *,
    router_url: str,
    question: str,
    func_doc: str,
    max_turns: int,
    max_steps_per_turn: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> List[List[str]]:
    """
    Minimal BFCL multi-turn loop: for each turn, repeatedly request a response;
    if the reply contains *decodable function calls*, keep adding steps; otherwise stop that turn.
    Next turn restarts with the *same* user prompt per BFCL’s protocol.
    Returns raw assistant strings for each step, grouped by turn.
    """
    convo_per_turn: List[List[str]] = []

    sys_prompt = (
        "You are a strictly function-calling assistant. "
        "Given a user request and a set of function tools, solve the task by emitting one or more function calls. "
        "Do not return natural language unless explicitly asked. "
        "If information is missing or a tool is unavailable, end the turn."
    )

    # Use simple OpenAI-style tools (pass as raw text in the user content so model sees docs);
    # If your model supports native tool schema, you can parse func_doc JSON and put it into 'tools' field instead.
    user_base = f"{question}\n\nAvailable function docs:\n{func_doc}"

    for _turn in range(max_turns):
        steps_this_turn: List[str] = []
        # Re-start the thread with the SAME user prompt (BFCL v3 design)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_base},
        ]
        for _step in range(max_steps_per_turn):
            reply = _chat_once(
                router_url=router_url,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            raw = _extract_raw_reply(reply)
            steps_this_turn.append(raw)

            # Check if the reply contains a decodable function call; if not, stop this turn
            if not _seems_function_call(reply):
                break

            # BFCL does not feed tool observations back between steps/turns in v3 evaluation protocol.
            # So we *don't* append tool outputs here; we simply continue sampling within the same turn
            # until model stops emitting tool calls. Then we start the next turn with the same user prompt.
        convo_per_turn.append(steps_this_turn)
    return convo_per_turn


def _chat_once(*, router_url: str, messages: List[Dict[str, str]], temperature: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
    payload = {
        "model": "slime-sglang",  # SLIME ignores this; SGLang routes to its engine
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        # Remove tool_choice as SGLang router doesn't support it
    }
    r = requests.post(router_url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if not data.get("choices"):
        raise RuntimeError(f"No choices from router: {data}")
    return data["choices"][0]["message"]


def _extract_raw_reply(message: Dict[str, Any]) -> str:
    """Return a raw string that BFCL handlers can later decode."""
    # If the model returned tool_calls in OpenAI format, keep that JSON
    if "tool_calls" in message and message["tool_calls"]:
        return json.dumps({"tool_calls": message["tool_calls"]}, ensure_ascii=False)
    # Else return the assistant content string
    return message.get("content", "")


def _seems_function_call(message: Dict[str, Any]) -> bool:
    if message.get("tool_calls"):
        return True
    txt = (message.get("content") or "").lower()
    # A loose heuristic for JSON-y function call stubs when not using tool_calls
    return ("\"function\"" in txt and "\"arguments\"" in txt) or re.search(r"\b[a-zA-Z_]\w*\s*\(", txt) is not None
