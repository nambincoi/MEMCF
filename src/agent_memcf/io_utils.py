import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from .models import AgentCFItemState, AgentCFUserState


def save_all_users_ranking_results(all_results: List[Dict], items_meta: Dict, output_file: str = "all_users_ranking_results.json"):
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    def get_item_info(item_id: str) -> Dict:
        if item_id in items_meta:
            info = items_meta[item_id]
            return {
                "item_id": item_id,
                "title": info.get("title", ""),
                "category": info.get("main_cat", "Unknown"),
            }
        return {"item_id": item_id, "title": f"Unknown Item {item_id}", "category": "Unknown"}

    final_results = []
    for res in all_results:
        user_result = {
            "user_id": res["user_id"],
            "num_candidates": len(res["candidates"]),
            "ground_truth_item_ids": res["ground_truth"],
            "candidate_item_ids": res["candidates"],
            "reranked_item_ids": res["predictions"],
            "candidate_items": [get_item_info(iid) for iid in res["candidates"]],
            "reranked_items": [get_item_info(iid) for iid in res["predictions"]],
            "metrics": res["metrics"],
            "baseline_metrics": res["baseline_metrics"],
        }
        final_results.append(user_result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved ranking results of {len(final_results)} users to {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def load_data(items_path: str, sequences_path: str, negatives_path: str):
    print("Loading data...")
    with open(items_path, "r", encoding="utf-8") as f:
        items_meta = json.load(f)
    with open(sequences_path, "r", encoding="utf-8") as f:
        user_sequences = json.load(f)
    with open(negatives_path, "r", encoding="utf-8") as f:
        user_negatives = json.load(f)
    print(f"Loaded {len(items_meta)} items")
    print(f"Loaded {len(user_sequences)} users")
    return items_meta, user_sequences, user_negatives


def calculate_recall_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    top_k = predictions[:k]
    hits = len(set(top_k) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0


def calculate_ndcg_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    top_k = predictions[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])
    return dcg / idcg if idcg > 0 else 0.0


def init_agentcf_item_states(items_meta: Dict[str, Dict[str, Any]]) -> Dict[str, AgentCFItemState]:
    item_states: Dict[str, AgentCFItemState] = {}
    for item_id, item_info in items_meta.items():
        title = item_info.get("title", f"Item {item_id}")
        category = item_info.get("main_cat", "Unknown")
        memory = f"The item is called '{title}'. The category is: '{category}'."
        item_states[item_id] = AgentCFItemState(item_id=item_id, title=title, category=category, memory=memory)
    return item_states


def get_or_create_user_state(user_states: Dict[str, AgentCFUserState], user_id: str) -> AgentCFUserState:
    if user_id not in user_states:
        user_states[user_id] = AgentCFUserState(user_id=user_id)
    return user_states[user_id]


def save_agent_states(
    user_states: Dict[str, AgentCFUserState], item_states: Dict[str, AgentCFItemState], filepath: str
) -> None:
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    data = {
        "user_states": {uid: asdict(state) for uid, state in user_states.items()},
        "item_states": {iid: asdict(state) for iid, state in item_states.items()},
        "metadata": {
            "num_users": len(user_states),
            "num_items": len(item_states),
            "save_timestamp": datetime.now().isoformat(),
        },
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Agent states saved to {filepath}")
    print(f"  - Users: {len(user_states)}")
    print(f"  - Items: {len(item_states)}")


def load_agent_states(filepath: str) -> Tuple[Dict[str, AgentCFUserState], Dict[str, AgentCFItemState]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agent state file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    loaded_user_states: Dict[str, AgentCFUserState] = {
        uid: AgentCFUserState(**state) for uid, state in data.get("user_states", {}).items()
    }
    loaded_item_states: Dict[str, AgentCFItemState] = {
        iid: AgentCFItemState(**state) for iid, state in data.get("item_states", {}).items()
    }
    print(f"✓ Agent states loaded from {filepath}")
    print(f"  - Users: {len(loaded_user_states)}")
    print(f"  - Items: {len(loaded_item_states)}")
    return loaded_user_states, loaded_item_states
