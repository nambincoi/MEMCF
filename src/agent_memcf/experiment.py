import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from .evaluation import evaluate_user
    from .io_utils import (
        get_or_create_user_state,
        init_agentcf_item_states,
        load_agent_states,
        load_data,
        save_agent_states,
        save_all_users_ranking_results,
    )
    from .memory_system import RecommendationMemorySystem, set_seed
    from .training import train_memory_from_fail_interactions
except ImportError:
    from evaluation import evaluate_user
    from io_utils import (
        get_or_create_user_state,
        init_agentcf_item_states,
        load_agent_states,
        load_data,
        save_agent_states,
        save_all_users_ranking_results,
    )
    from memory_system import RecommendationMemorySystem, set_seed
    from training import train_memory_from_fail_interactions


set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment configuration")
    parser.add_argument("--data_name", type=str, default="Video_Game")
    parser.add_argument("--use_memory", action="store_true", default=True)
    parser.add_argument("--LOAD_SAVED_MEMORY", action="store_true", default=False)
    parser.add_argument("--wo_evolving", action="store_true", default=True)
    parser.add_argument("--wo_link", action="store_true", default=False)
    parser.add_argument("--max_evolutions_per_memory", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--link_size", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=1)
    parser.add_argument("--k_memories", type=int, default=1)
    parser.add_argument("--eval_variants", type=str, default="both", help="v1, v2, or both")
    parser.add_argument("--fewshot_ranking", action="store_true", default=False)
    parser.add_argument("--k_shot", type=int, default=3)
    parser.add_argument("--number_of_users", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    data_name = args.data_name
    use_memory = args.use_memory
    load_saved_memory = args.LOAD_SAVED_MEMORY
    wo_evolving = args.wo_evolving
    wo_link = args.wo_link
    max_evolutions_per_memory = args.max_evolutions_per_memory
    link_size = args.link_size
    max_iterations = args.max_iterations
    k_memories = args.k_memories
    fewshot_ranking = args.fewshot_ranking
    k_shot = args.k_shot
    number_of_users = args.number_of_users

    eval_variants_raw = args.eval_variants.lower().strip()
    if eval_variants_raw == "both":
        eval_variants = ["v1", "v2"]
    else:
        eval_variants = [v.strip() for v in eval_variants_raw.split(",") if v.strip()]
    valid_variants = {"v1", "v2"}
    if not eval_variants or any(v not in valid_variants for v in eval_variants):
        raise ValueError(f"Invalid --eval_variants={args.eval_variants}. Use v1, v2, or both.")

    base_dir = os.getenv("AGENTICREC_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
    data_root = os.getenv("AGENTICREC_DATA_ROOT", os.path.join(base_dir, "data"))
    eval_root = os.getenv("AGENTICREC_EVAL_ROOT", os.path.join(base_dir, "evaluation_results"))
    memory_root = os.getenv("AGENTICREC_MEMORY_ROOT", os.path.join(base_dir, "agent_memory"))

    items_path = os.path.join(data_root, data_name, "items.json")
    sequences_path = os.path.join(data_root, data_name, "user_sequences_10.json")
    negatives_path = os.path.join(data_root, data_name, "user_negatives_10.json")

    if use_memory:
        if wo_evolving:
            output_base = f"nuser{number_of_users}_fail_interactions_no_evolving_k{k_memories}_iter{max_iterations}_memory"
            memory_file_path = os.path.join(
                memory_root, data_name, f"nuser{number_of_users}_fail_interactions_no_evolving_iter{max_iterations}.json"
            )
        elif wo_link:
            output_base = (
                f"nuser{number_of_users}_fail_interactions_no_link_k{k_memories}_iter{max_iterations}_"
                f"memory_maxevolution{str(max_evolutions_per_memory)}"
            )
            memory_file_path = os.path.join(
                memory_root,
                data_name,
                f"nuser{number_of_users}_fail_interactions_no_link_iter{max_iterations}_maxevolution{str(max_evolutions_per_memory)}.json",
            )
        else:
            output_base = (
                f"nuser{number_of_users}_global_fail_interactions_{k_memories}_iter{max_iterations}_"
                f"link{link_size}_memory_maxevolution{str(max_evolutions_per_memory)}"
            )
            memory_file_path = os.path.join(
                memory_root,
                data_name,
                f"nuser{number_of_users}_global_fail_interactions_iter{max_iterations}_link{link_size}_maxevolution{str(max_evolutions_per_memory)}.json",
            )
    else:
        if fewshot_ranking:
            output_base = f"nuser{number_of_users}_fewshot_{k_shot}_users_ranking_no_memory"
        else:
            output_base = f"nuser{number_of_users}_zeroshot_users_ranking_no_memory"

    items_meta, user_sequences, user_negatives = load_data(items_path, sequences_path, negatives_path)
    print(f"Total users loaded: {len(user_sequences)}")

    user_ids = list(user_sequences.keys())[:number_of_users]
    if not use_memory and fewshot_ranking:
        sample_user_ids = list(user_sequences.keys())[number_of_users:]

    global_memory = RecommendationMemorySystem(use_gemini_embeddings=True)
    user_states = {}
    item_states = init_agentcf_item_states(items_meta)
    state_file_path = os.path.join(memory_root, data_name, f"nuser{number_of_users}_agent_states_iter{max_iterations}.json")

    if use_memory:
        if load_saved_memory and os.path.exists(memory_file_path):
            print("\n" + "=" * 80)
            print("LOADING SAVED MEMORY SYSTEM")
            print("=" * 80)
            global_memory.load_memory(memory_file_path)
            if os.path.exists(state_file_path):
                user_states, loaded_item_states = load_agent_states(state_file_path)
                if loaded_item_states:
                    item_states = loaded_item_states
            else:
                print(f"⚠ No saved agent states found at {state_file_path}. Using fresh user/item memories.")
        else:
            print("\n" + "=" * 80)
            print("PHASE 1: TRAINING WITH CROSS-USER EVOLVING ONLY")
            print("=" * 80)

            global_memory = RecommendationMemorySystem(use_gemini_embeddings=False)

            for user_id in tqdm(user_ids, desc="Cross-user Training"):
                user_data = user_sequences[user_id]
                print(f"\nProcessing user {user_id} ({len(user_data['train'])} interactions)")

                temp_system = RecommendationMemorySystem.__new__(RecommendationMemorySystem)
                temp_system.llm_name = global_memory.llm_name
                temp_system.embedding_model_name = global_memory.embedding_model_name
                temp_system.chat_model_name = global_memory.chat_model_name
                temp_system.chat_api_base = global_memory.chat_api_base
                temp_system.embedding_api_base = global_memory.embedding_api_base
                temp_system.api_key = global_memory.api_key
                temp_system.use_api_chat = global_memory.use_api_chat
                temp_system.use_api_embedding = global_memory.use_api_embedding
                temp_system.model = getattr(global_memory, "model", None)
                temp_system.tokenizer = getattr(global_memory, "tokenizer", None)
                temp_system.embedding_model = getattr(global_memory, "embedding_model", None)
                temp_system.behavior_memories = []
                temp_system.user_interaction_history = []
                temp_system.next_thought_id = 0

                try:
                    new_memories = train_memory_from_fail_interactions(
                        user_id=user_id,
                        user_data=user_data,
                        memory_system=temp_system,
                        user_states=user_states,
                        item_states=item_states,
                        max_iterations=max_iterations,
                    )
                except Exception as e:
                    print(f"\nError evaluating user {user_id}: {e}")
                    continue
                if not new_memories:
                    print("  → No new memories generated, skipping...")
                    continue

                print(f"  → Generated {len(new_memories)} fail-interaction memories")
                max_global_id = global_memory.next_thought_id - 1 if global_memory.behavior_memories else -1

                for new_mem in new_memories:
                    new_mem.thought_id += max_global_id + 1
                    if not wo_evolving:
                        try:
                            linked_ids = global_memory.link_behavior_memories(new_mem, k=link_size, wo_link=wo_link)
                            new_mem.links = linked_ids
                            global_memory.evolve_behavior_memories(
                                new_mem,
                                linked_ids,
                                max_evolutions_per_memory=max_evolutions_per_memory,
                            )
                        except Exception as e:
                            print(f"\nError evolving memory for user {user_id}: {e}")

                    global_memory.behavior_memories.append(new_mem)
                    global_memory.next_thought_id = new_mem.thought_id + 1

                print(f"  → Global memory pool now has {len(global_memory.behavior_memories)} memories")
                global_memory.save_memory(memory_file_path, format="json")
                save_agent_states(user_states, item_states, state_file_path)
                print(f"  → Overwritten common global memory file: {memory_file_path}")

        print("\n" + "=" * 80)
        print("SAVING GLOBAL CROSS-USER EVOLVING MEMORY")
        print("=" * 80)
        global_memory.save_memory(memory_file_path, format="json")
        save_agent_states(user_states, item_states, state_file_path)
        global_memory.print_evolution_report()
        stats = global_memory.get_evolution_statistics()
        stats_file = memory_file_path.replace(".json", "_evolution_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Evolution statistics saved to {stats_file}")
    elif load_saved_memory:
        if os.path.exists(state_file_path):
            user_states, loaded_item_states = load_agent_states(state_file_path)
            if loaded_item_states:
                item_states = loaded_item_states
        else:
            print(f"⚠ LOAD_SAVED_MEMORY is enabled but no saved agent states at {state_file_path}")

    print("\n" + "=" * 80)
    print("PHASE 2: VALIDATION SET EVALUATION")
    print("=" * 80)

    baseline_metrics = {
        "recall@5": [],
        "recall@10": [],
        "recall@20": [],
        "ndcg@5": [],
        "ndcg@10": [],
        "ndcg@20": [],
    }
    val_metrics_by_variant = {
        variant: {"recall@5": [], "recall@10": [], "recall@20": [], "ndcg@5": [], "ndcg@10": [], "ndcg@20": []}
        for variant in eval_variants
    }
    all_user_results_by_variant = {variant: [] for variant in eval_variants}

    if not use_memory:
        global_memory = RecommendationMemorySystem(use_gemini_embeddings=True)
    for user_id in tqdm(user_ids, desc="Validation"):
        try:
            user_data = user_sequences[user_id]
            negative_data = user_negatives.get(user_id, {})
            fixed_candidates = user_data["test"] + negative_data.get("test_neg", [])
            random.shuffle(fixed_candidates)

            sample_user_id_list = random.sample(sample_user_ids, k_shot) if not use_memory and fewshot_ranking else None
            if sample_user_id_list is not None:
                sample_user_list = []
                negative_data_sample_list = []
                for sid in sample_user_id_list:
                    sample_user_list.append(user_sequences[sid] if sid else None)
                    negative_data_sample_list.append(user_negatives.get(sid, {}) if sid else None)
            else:
                sample_user_list = None
                negative_data_sample_list = None

            baseline_added = False
            for variant in eval_variants:
                baseline_metric, metrics, candidates, predictions, ground_truth = evaluate_user(
                    user_data,
                    negative_data,
                    items_meta,
                    global_memory,
                    eval_type="test",
                    use_memory=use_memory,
                    k_memories=k_memories,
                    sample_user_list=sample_user_list,
                    negative_data_sample_list=negative_data_sample_list,
                    user_state=get_or_create_user_state(user_states, user_id),
                    item_states=item_states,
                    eval_variant=variant,
                    fixed_candidates=fixed_candidates,
                )
                all_user_results_by_variant[variant].append(
                    {
                        "user_id": user_id,
                        "ground_truth": ground_truth,
                        "candidates": candidates,
                        "predictions": predictions,
                        "metrics": metrics,
                        "baseline_metrics": baseline_metric,
                    }
                )
                for key in val_metrics_by_variant[variant]:
                    val_metrics_by_variant[variant][key].append(metrics[key])
                if not baseline_added:
                    for key in baseline_metrics:
                        baseline_metrics[key].append(baseline_metric[key])
                    baseline_added = True
        except Exception as e:
            print(f"\nError evaluating user {user_id}: {e}")
            continue

    os.makedirs(os.path.join(eval_root, data_name), exist_ok=True)
    for variant in eval_variants:
        output_file = os.path.join(eval_root, data_name, f"{output_base}_{variant}.json")
        save_all_users_ranking_results(
            all_results=all_user_results_by_variant[variant],
            items_meta=items_meta,
            output_file=output_file,
        )

    print("\nValidation Results:")
    print("-" * 80)
    for metric in ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]:
        if len(baseline_metrics[metric]) > 0:
            mean_val = np.mean(baseline_metrics[metric])
            print(f"Baseline {metric:10s}: {mean_val:.4f}")
        else:
            print(f"Baseline {metric:10s}: N/A")
    for variant in eval_variants:
        print("-" * 80)
        print(f"Variant {variant.upper()}:")
        for metric in ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]:
            if len(val_metrics_by_variant[variant][metric]) > 0:
                mean_val = np.mean(val_metrics_by_variant[variant][metric])
                print(f"{metric:12s}: {mean_val:.4f}")
            else:
                print(f"{metric:12s}: N/A")

    summary = {
        "model": "agent_memcf",
        "dataset": data_name,
        "number_of_users": number_of_users,
        "max_iterations": max_iterations,
        "k_memories": k_memories,
        "use_memory": use_memory,
        "eval_variants": eval_variants,
        "baseline_metrics": {
            metric: (float(np.mean(baseline_metrics[metric])) if len(baseline_metrics[metric]) > 0 else None)
            for metric in ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
        },
        "variant_metrics": {
            variant: {
                metric: (
                    float(np.mean(val_metrics_by_variant[variant][metric]))
                    if len(val_metrics_by_variant[variant][metric]) > 0
                    else None
                )
                for metric in ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
            }
            for variant in eval_variants
        },
    }
    summary_path = os.path.join(eval_root, data_name, f"{output_base}.summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved AgentMemCF summary to {summary_path}")


if __name__ == "__main__":
    main()
