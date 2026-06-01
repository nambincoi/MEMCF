import json
import random
from typing import Dict, List, Optional

from .io_utils import calculate_ndcg_at_k, calculate_recall_at_k
from .memory_system import RecommendationMemorySystem
from .models import AgentCFItemState, AgentCFUserState


def evaluate_user(
    user_data: Dict,
    negative_data: Dict,
    items_meta: Dict,
    memory_system: RecommendationMemorySystem,
    eval_type: str = "test",
    use_memory=True,
    k_memories: int = 5,
    sample_user_list: List = None,
    negative_data_sample_list: List = None,
    user_state: Optional[AgentCFUserState] = None,
    item_states: Optional[Dict[str, AgentCFItemState]] = None,
    eval_variant: str = "v2",
    fixed_candidates: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate for a single user with LLM-based ranking."""
    if eval_type == "val":
        ground_truth = user_data["val"]
        negatives = negative_data.get("val_neg", [])
    else:
        ground_truth = user_data["test"]
        negatives = negative_data.get("test_neg", [])

    if sample_user_list is not None:
        ground_truth_sample_fewshot = []
        negatives_sample_fewshot = []
        for i in range(len(sample_user_list)):
            sample_user_data = sample_user_list[i]
            negative_data_sample = negative_data_sample_list[i]
            ground_truth_sample = sample_user_data.get("val", [])
            ground_truth_sample_fewshot.append(ground_truth_sample)
            negatives_sample = negative_data_sample.get("val_neg", [])
            negatives_sample_fewshot.append(negatives_sample)

    train_items_info = []
    user_profile_texts = []
    for item_id in user_data["train"][-10:]:
        item_state = item_states.get(item_id) if item_states else None
        if item_state is not None:
            title = item_state.title
            category = item_state.category
        elif item_id in items_meta:
            item_info = items_meta[item_id]
            title = item_info.get("title", "")
            category = item_info.get("main_cat", "Unknown")
        else:
            title = f"Item {item_id}"
            category = "Unknown"

        train_items_info.append({"item_id": item_id, "title": title, "category": category})
        user_profile_texts.append(f"{title} {category}")

    user_profile_text = " ".join(user_profile_texts)

    prompt_sample = ""
    if sample_user_list is not None:
        prompt_sample = "Learn from the following examples:\n"
        for i in range(len(sample_user_list)):
            sample_user_data = sample_user_list[i]
            sample_train_items_info = []
            for item_id in sample_user_data["train"][-10:]:
                if item_id in items_meta:
                    item_info = items_meta[item_id]
                    title = item_info.get("title", "")
                    category = item_info.get("main_cat", "Unknown")
                    sample_train_items_info.append({"item_id": item_id, "title": title, "category": category})
            sample_user_profile_texts = []
            for item in sample_train_items_info:
                sample_user_profile_texts.append(f"{item['title']} {item['category']}")
            sample_user_profile_text = " ".join(sample_user_profile_texts)

            candidates_sample = ground_truth_sample_fewshot[i] + negatives_sample_fewshot[i]
            random.shuffle(candidates_sample)
            candidate_items_info_sample = []
            for item_id in candidates_sample:
                if item_id in items_meta:
                    item_info = items_meta[item_id]
                    candidate_items_info_sample.append(
                        {
                            "item_id": item_id,
                            "title": item_info.get("title", ""),
                            "category": item_info.get("main_cat", "Unknown"),
                        }
                    )
                else:
                    candidate_items_info_sample.append(
                        {"item_id": item_id, "title": f"Item {item_id}", "category": "Unknown"}
                    )
            prompt_sample += f"""
            Example {i+1}:
            Other user Recent History: {sample_user_profile_text}
            Candidate Items: {json.dumps(candidate_items_info_sample, indent=2)}
            You should set the true items "{json.dumps(ground_truth_sample_fewshot[i], indent=2)}" at the top of the ranking.\n
            """

    if fixed_candidates is not None:
        candidates = list(fixed_candidates)
    else:
        candidates = ground_truth + negatives
        random.shuffle(candidates)

    candidate_items_info = []
    for item_id in candidates:
        item_state = item_states.get(item_id) if item_states else None
        if eval_variant == "v2" and item_state is not None:
            candidate_items_info.append(
                {
                    "item_id": item_id,
                    "title": item_state.title,
                    "category": item_state.category,
                    "memory": item_state.memory,
                }
            )
        elif eval_variant == "v2" and item_id in items_meta:
            item_info = items_meta[item_id]
            title = item_info.get("title", "")
            category = item_info.get("main_cat", "Unknown")
            candidate_items_info.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "category": category,
                    "memory": f"The item is called '{title}'. The category is: '{category}'.",
                }
            )
        elif eval_variant == "v2":
            candidate_items_info.append(
                {"item_id": item_id, "title": f"Item {item_id}", "category": "Unknown", "memory": "No item memory available."}
            )
        elif item_state is not None:
            candidate_items_info.append({"item_id": item_id, "title": item_state.title, "category": item_state.category})
        elif item_id in items_meta:
            item_info = items_meta[item_id]
            candidate_items_info.append(
                {
                    "item_id": item_id,
                    "title": item_info.get("title", ""),
                    "category": item_info.get("main_cat", "Unknown"),
                }
            )
        else:
            candidate_items_info.append({"item_id": item_id, "title": f"Item {item_id}", "category": "Unknown"})

    if use_memory:
        retrieved_memories = memory_system.retrieve_relevant_memories(user_profile_text, k=k_memories)
    else:
        retrieved_memories = None

    predictions = memory_system.llm_ranking(
        train_items_info,
        candidate_items_info,
        retrieved_memories,
        prompt_sample,
        user_memory=user_state.short_term_memory if (eval_variant == "v2" and user_state) else None,
        eval_variant=eval_variant,
    )
    baseline_metric = {
        "recall@5": calculate_recall_at_k(candidates, ground_truth, 5),
        "recall@10": calculate_recall_at_k(candidates, ground_truth, 10),
        "recall@20": calculate_recall_at_k(candidates, ground_truth, 20),
        "ndcg@5": calculate_ndcg_at_k(candidates, ground_truth, 5),
        "ndcg@10": calculate_ndcg_at_k(candidates, ground_truth, 10),
        "ndcg@20": calculate_ndcg_at_k(candidates, ground_truth, 20),
    }
    metrics = {
        "recall@5": calculate_recall_at_k(predictions, ground_truth, 5),
        "recall@10": calculate_recall_at_k(predictions, ground_truth, 10),
        "recall@20": calculate_recall_at_k(predictions, ground_truth, 20),
        "ndcg@5": calculate_ndcg_at_k(predictions, ground_truth, 5),
        "ndcg@10": calculate_ndcg_at_k(predictions, ground_truth, 10),
        "ndcg@20": calculate_ndcg_at_k(predictions, ground_truth, 20),
    }
    return baseline_metric, metrics, candidates, predictions, ground_truth
