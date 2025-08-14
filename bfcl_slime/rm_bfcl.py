# bfcl_slime/rm_bfcl.py
import os, json, importlib
from typing import List, Dict, Any

# SLIME will import bfcl_reward via --custom-rm-path bfcl_slime.rm_bfcl:bfcl_reward
def bfcl_reward(samples: List[Dict[str, Any]]) -> List[float]:
    """
    samples[i]["metadata"] contains:
        {
          "id": <bfcl_id>,
          "result": [[raw_step_str, ...], [ ... turn2 ... ], ...],
          "bfcl_meta": {"category": "...", "handler_name": "...", "question": "...", "func_doc": "..."}
        }
    Returns one reward per sample: 1.0 if passes BFCL checker, else 0.0.
    """
    # Lazy import bfcl-eval only here (worker process)
    from bfcl_eval.model_handler.handler_map import handler_map  # official handlers
    from bfcl_eval.eval_checker import eval_runner_helper as erh      # has multi_turn_checker
    # Load ground-truth "possible answers" for the category
    gt_loader = _ground_truth_loader()

    rewards: List[float] = []
    for s in samples:
        meta = s["metadata"]
        bfcl_id = meta["id"]
        result = meta["result"]
        cat = meta["bfcl_meta"]["category"]
        handler_name = meta["bfcl_meta"].get("handler_name", "openai")

        # Build a handler for decoding model outputs into executable calls
        HandlerCls = handler_map[handler_name]
        handler = HandlerCls()

        # Decode each step using the handler's exec decoder
        decoded_turns: List[List[str]] = []
        for single_turn in result:
            decoded_steps = []
            for raw in single_turn:
                try:
                    decoded = handler.decode_execute(raw)  # returns a list or a single string; normalize to str
                    if isinstance(decoded, list):
                        decoded_steps.extend(decoded)
                    else:
                        decoded_steps.append(decoded)
                except Exception:
                    # If cannot decode, mark as empty -> BFCL will end the turn
                    decoded_steps.append("")
            decoded_turns.append(decoded_steps)

        # Fetch ground truth and the full test entry (question+func_doc) for this id
        test_entry = gt_loader.load_test_entry(bfcl_id, cat)
        gt_multi_turn = gt_loader.load_ground_truth(bfcl_id, cat)

        # Official multi-turn checker (returns per-turn validity and an overall flag)
        # Signature in bfcl-eval mirrors the public evaluator
        check = erh.multi_turn_checker(
            decoded_turns,         # model-decoded executable calls
            gt_multi_turn,         # list of acceptable answers
            test_entry,            # full test entry (question, func_doc, etc.)
            cat,
            handler_name,
        )

        rewards.append(1.0 if bool(check.get("overall_valid", False)) else 0.0)
    return rewards


# -----------------------------
# Ground-truth loader
# -----------------------------
class _ground_truth_loader:
    def __init__(self):
        import importlib.resources as ir
        self.pkg = "bfcl_eval"
        self.data_dir = ir.files(self.pkg) / "data"
        self.possible_dir = ir.files(self.pkg) / "possible_answer"

    def load_test_entry(self, bfcl_id: str, category: str) -> Dict[str, Any]:
        # BFCL dataset files are named BFCL_v3_<category>.json(.l)
        path = None
        for ext in (".json", ".jsonl"):
            p = self.data_dir / f"BFCL_v3_{category}{ext}"
            if p.exists():
                path = p
                break
        if path is None:
            raise FileNotFoundError(f"BFCL_v3_{category} not found inside bfcl-eval package.")
        # Scan for the entry (files are not huge)
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["id"] == bfcl_id:
                        return rec
        else:
            data = json.load(open(path, "r", encoding="utf-8"))
            for rec in data:
                if rec["id"] == bfcl_id:
                    return rec
        raise KeyError(f"{bfcl_id} not in {path.name}")

    def load_ground_truth(self, bfcl_id: str, category: str) -> List[Any]:
        # possible answers file BFCL_v3_<category>.json in /possible_answer
        p = self.possible_dir / f"BFCL_v3_{category}.json"
        if not p.exists():
            raise FileNotFoundError(f"Possible answers file missing: {p}")
        data = json.load(open(p, "r", encoding="utf-8"))
        # some files map id->answers; others list; handle both
        if isinstance(data, dict) and bfcl_id in data:
            return data[bfcl_id]
        if isinstance(data, list):
            for rec in data:
                if rec.get("id") == bfcl_id:
                    return rec.get("answers", rec.get("answer", []))
        raise KeyError(f"Possible answers for {bfcl_id} not found.")
