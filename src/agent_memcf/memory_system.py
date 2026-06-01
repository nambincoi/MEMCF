import json
import os
import pickle
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .models import BehaviorMemory, UserInteraction


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


set_seed(42)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: transformers/torch not available: {e}")
    print("Will use OpenAI-compatible API endpoints if provided via env vars.")
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: sentence_transformers not available: {e}")
    print("Will use API embeddings or hash fallback")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class RecommendationMemorySystem:
    """A-Mem adapted for Amazon product recommendation with Memory Evolution"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gemini_embeddings: bool = None,
        chat_api_base: Optional[str] = None,
        embedding_api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        chat_model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
    ):
        _ = use_gemini_embeddings

        self.llm_name = model_name
        self.embedding_model_name = embedding_model_name or os.getenv("embedding_model_name") or embedding_model
        self.chat_model_name = chat_model_name or os.getenv("chat_model_name") or model_name
        self.chat_api_base = (chat_api_base or os.getenv("chat_api_base") or os.getenv("api_base") or "").rstrip("/")
        self.embedding_api_base = (
            embedding_api_base or os.getenv("embedding_api_base") or os.getenv("api_base") or ""
        ).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"

        self.use_api_chat = bool(self.chat_api_base)
        self.use_api_embedding = False

        self.tokenizer = None
        self.model = None
        self.embedding_model = None

        if not self.use_api_chat:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError(
                    "Local chat model requires transformers+torch, or set chat_api_base/api_base env vars."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                dtype=torch.float16,
                device_map="auto",
            )

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        else:
            raise RuntimeError(
                "sentence_transformers is required for VIRAL-style embedding. "
                "Please install sentence-transformers."
            )

        self.behavior_memories: List[BehaviorMemory] = []
        self.user_interaction_history: List[UserInteraction] = []
        self.next_thought_id = 0

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API HTTP {e.code} at {url}: {body}") from e

    def qwen_generate(self, prompt: str, role_prompt="You are a helpful AI assistant.", max_new_tokens=8000) -> str:
        if self.use_api_chat:
            endpoint = f"{self.chat_api_base}/chat/completions"
            payload = {
                "model": self.chat_model_name,
                "messages": [
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_new_tokens,
                "temperature": 0.2,
            }
            result = self._post_json(endpoint, payload)
            content = result["choices"][0]["message"]["content"]
            if isinstance(content, list):
                return "".join(chunk.get("text", "") for chunk in content if isinstance(chunk, dict))
            return str(content)

        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        gen_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def _create_embedding(self, text: str) -> np.ndarray:
        emb = self.embedding_model.encode(str(text), convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)

    def _simple_hash_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def add_interaction(
        self,
        item_id: str,
        item_name: str,
        item_category: str,
        action_type: str = "purchase",
        rating: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> UserInteraction:
        interaction = UserInteraction(
            item_id=item_id,
            item_name=item_name,
            item_category=item_category,
            action_type=action_type,
            rating=rating,
            metadata=metadata or {},
        )
        self.user_interaction_history.append(interaction)
        return interaction

    def create_behavior_thought(
        self, interaction_window: List[UserInteraction], k_neighbors: int = 10
    ) -> BehaviorMemory:
        interaction_summary = []
        for interaction in interaction_window:
            summary = {
                "item": interaction.item_name,
                "category": interaction.item_category,
                "action": interaction.action_type,
            }
            interaction_summary.append(summary)

        prompt = f"""Analyze this failed recommendation interaction.
        Input: {json.dumps(interaction_summary, indent=2)}

        Context:
        - The interaction contains a wrong choice and the preferred correct item.
        - Your job is to capture why the wrong choice happened and what correction rule should be applied next time.

        Return ONLY a JSON object in this format:
        {{
        "behavior_explanation": "2-3 concise sentences explaining why the wrong choice was made versus the correct item",
        "pattern_description": "2-3 concise sentences describing a correction pattern/rule for future ranking",
        "keywords": ["kw1", "kw2", ...] (5 short fail-interaction signals, such as mismatch types or decisive attributes)
        }}

        Requirements:
        - Ground every statement in the input interaction.
        - Emphasize contrast between wrong and correct choice.
        - Avoid generic shopping summaries."""

        try:
            response = self.qwen_generate(prompt=prompt, role_prompt="You are a behavioral memory modeling system.")
            result_text = response.strip()

            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)

            behavior_explanation = result.get("behavior_explanation", "")
            pattern_description = result.get("pattern_description", "")
            keywords = result.get("keywords", [])
        except Exception as e:
            print(f"Error in behavior analysis: {e}")
            behavior_explanation = f"A failed interaction occurred with {len(interaction_window)} compared items."
            pattern_description = "Correction rule is unclear; prefer signals from the preferred item over the wrong choice."
            keywords = [i.item_category for i in interaction_window[:3]]

        combined_text = f"{behavior_explanation} {pattern_description} {' '.join(keywords)}"
        embedding = self._create_embedding(combined_text)

        behavior_memory = BehaviorMemory(
            thought_id=self.next_thought_id,
            interaction_sequence=interaction_window.copy(),
            behavior_explanation=behavior_explanation,
            pattern_description=pattern_description,
            keywords=keywords,
            embedding=embedding,
        )

        self.next_thought_id += 1
        return behavior_memory

    def link_behavior_memories(self, new_memory: BehaviorMemory, k: int = 5, wo_link=False) -> List[int]:
        if len(self.behavior_memories) == 0:
            return []

        similarities = []
        for memory in self.behavior_memories:
            sim = self._cosine_similarity(new_memory.embedding, memory.embedding)
            similarities.append((memory.thought_id, sim, memory))

        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_k = similarities[: min(k, len(similarities))]

        if len(nearest_k) == 0:
            return []
        if wo_link:
            return [thought_id for thought_id, _, _ in nearest_k]

        nearest_info = []
        for thought_id, sim, memory in nearest_k:
            nearest_info.append(
                {
                    "thought_id": thought_id,
                    "behavior_explanation": memory.behavior_explanation,
                    "pattern": memory.pattern_description,
                    "similarity": float(sim),
                }
            )

        prompt = f"""Determine if the new fail-interaction memory should be linked to past fail memories.
        New Pattern:
        - Behavior: {new_memory.behavior_explanation}
        - Pattern: {new_memory.pattern_description}

        Similar Past Patterns:
        {json.dumps(nearest_info, indent=2)}

        Link ONLY if:
        - They share a similar error/correction pattern (same mismatch type or same correction signal).
        - They imply a consistent fix strategy across users or interactions.
        - Their wrong-vs-correct contrast is semantically aligned.
        Do NOT link if they describe unrelated failure reasons.

        Return JSON:
        {{
        "should_link": true/false,
        "linked_thought_ids": [list of IDs],
        "reasoning": "1-2 sentences explaining shared fail/correction evidence"
        }}
        Keep reasoning concise and specific."""

        try:
            response = self.qwen_generate(prompt=prompt, role_prompt="You are a behavioral memory modeling system.")
            result_text = response.strip()

            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            if result.get("should_link", False):
                return result.get("linked_thought_ids", [])
            return []
        except Exception as e:
            print(f"Error in linking: {e}")
            return [thought_id for thought_id, sim, _ in nearest_k if sim > 0.65]

    def evolve_behavior_memories(
        self, new_memory: BehaviorMemory, linked_ids: List[int], max_evolutions_per_memory: Optional[int] = None
    ) -> None:
        if len(linked_ids) == 0:
            return

        linked_memories = [m for m in self.behavior_memories if m.thought_id in linked_ids]
        if len(linked_memories) == 0:
            return

        evolvable_memories = []
        for mem in linked_memories:
            if max_evolutions_per_memory is not None:
                mem.max_evolutions = max_evolutions_per_memory

            if mem.can_evolve():
                evolvable_memories.append(mem)
            else:
                print(f"  ⚠ Memory {mem.thought_id} reached max evolutions ({mem.evolution_count}), skipping...")

        if len(evolvable_memories) == 0:
            print("  → No memories available for evolution (all reached max)")
            return

        mem_info = []
        for mem in evolvable_memories:
            mem_info.append(
                {
                    "thought_id": mem.thought_id,
                    "behavior_explanation": mem.behavior_explanation,
                    "pattern": mem.pattern_description,
                    "evolution_count": mem.evolution_count,
                }
            )

        prompt = f"""Determine if past fail memories should be updated using a new fail case.
        New Pattern:
        - Behavior: {new_memory.behavior_explanation}
        - Pattern: {new_memory.pattern_description}

        Linked Past Patterns (with evolution history):
        {json.dumps(mem_info, indent=2)}

        Update Guidelines:
        - Update when the new fail case provides clearer correction evidence for an existing fail pattern.
        - Refine wording toward a stronger wrong-vs-correct contrast.
        - Prefer updates that improve future error avoidance rules.
        - Skip updates when the new fail case is unrelated.

        Return JSON:
        {{
        "should_evolve": true/false,
        "updates": [
            {{
            "thought_id": ID,
            "behavior_explanation": "updated text or null",
            "new_pattern": "updated text or null",
            "reasoning": "1 sentence explaining how the fail-correction rule is refined"
            }}
        ]
        }}
        Ensure updates are grounded in input data and reasoning is concise."""

        try:
            response = self.qwen_generate(prompt=prompt, role_prompt="You are a behavioral memory modeling system.")
            result_text = response.strip()

            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)

            if result.get("should_evolve", False):
                updates = result.get("updates", [])
                for update in updates:
                    thought_id = update.get("thought_id")
                    memory = next((m for m in self.behavior_memories if m.thought_id == thought_id), None)
                    if memory:
                        old_values = {
                            "behavior_explanation": memory.behavior_explanation,
                            "pattern_description": memory.pattern_description,
                        }

                        updated = False
                        update_type = []
                        if update.get("behavior_explanation"):
                            memory.behavior_explanation = update["behavior_explanation"]
                            updated = True
                            update_type.append("behavior_explanation")

                        if update.get("new_pattern"):
                            memory.pattern_description = update["new_pattern"]
                            updated = True
                            update_type.append("pattern")

                        if updated:
                            combined_text = (
                                f"{memory.behavior_explanation} {memory.pattern_description} {' '.join(memory.keywords)}"
                            )
                            memory.embedding = self._create_embedding(combined_text)
                            new_values = {
                                "behavior_explanation": memory.behavior_explanation,
                                "pattern_description": memory.pattern_description,
                            }
                            memory.record_evolution(
                                update_type=", ".join(update_type),
                                old_values=old_values,
                                new_values=new_values,
                                reasoning=update.get("reasoning", ""),
                            )
        except Exception as e:
            print(f"Error in memory evolution: {e}")

    def add_behavior_memory(self, interaction_window: List[UserInteraction], k_neighbors: int = 5) -> BehaviorMemory:
        behavior_memory = self.create_behavior_thought(interaction_window, k_neighbors)
        linked_ids = self.link_behavior_memories(behavior_memory, k_neighbors)
        behavior_memory.links = linked_ids

        for thought_id in linked_ids:
            memory = next((m for m in self.behavior_memories if m.thought_id == thought_id), None)
            if memory and behavior_memory.thought_id not in memory.links:
                memory.links.append(behavior_memory.thought_id)

        self.evolve_behavior_memories(behavior_memory, linked_ids)
        self.behavior_memories.append(behavior_memory)
        return behavior_memory

    def retrieve_relevant_memories(self, user_profile_text: str, k: int = 5) -> List[BehaviorMemory]:
        if len(self.behavior_memories) == 0:
            return []

        profile_embedding = self._create_embedding(user_profile_text)
        similarities = []
        for memory in self.behavior_memories:
            sim = self._cosine_similarity(profile_embedding, memory.embedding)
            similarities.append((memory, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in similarities[: min(k, len(similarities))]]

    def llm_ranking(
        self,
        train_items: List[Dict],
        candidate_items: List[Dict],
        retrieved_memories: List[BehaviorMemory],
        prompt_sample: str,
        user_memory: Optional[str] = None,
        eval_variant: str = "v2",
    ) -> List[str]:
        """Use LLM to rank candidate items based on user profile, item memories, and behavior memories."""
        user_profile = []
        for item in train_items:
            user_profile.append(
                {"title": item["title"], "category": item["category"], "item_id": item["item_id"]}
            )

        candidate_info = []
        for item in candidate_items:
            if eval_variant == "v2":
                candidate_info.append(
                    {
                        "item_id": item["item_id"],
                        "title": item["title"],
                        "category": item["category"],
                        "memory": item.get("memory", ""),
                    }
                )
            else:
                candidate_info.append(
                    {"item_id": item["item_id"], "title": item["title"], "category": item["category"]}
                )
        effective_user_memory = user_memory or "I enjoy discovering new items."

        if retrieved_memories is not None:
            memory_thoughts = []
            for mem in retrieved_memories:
                memory_thoughts.append(
                    {
                        "behavior_explanation": mem.behavior_explanation,
                        "pattern": mem.pattern_description,
                    }
                )
            if eval_variant == "v2":
                prompt = f"""
            You are ranking candidate items for a user based on user memory, candidate item memory, shopping history, and fail-correction memories.

            Inputs:
            User Self Memory:
            {effective_user_memory}

            User Recent History (prioritize most recent):
            {json.dumps(user_profile[-10:], indent=2)}

            Collaborative Memory Insights (known wrong-choice patterns and correction rules):
            {json.dumps(memory_thoughts, indent=2)}

            Candidate Items:
            {json.dumps(candidate_info, indent=2)}

            Ranking Rules:
            - Prioritize candidates matching user self memory and recent interaction history.
            - Use candidate item memory text to differentiate close candidates.
            - Avoid repeating known wrong-choice patterns captured in memory.
            - Prefer candidates consistent with correction patterns from memory.
            - Ensure category consistency with history and correction signals.
            - If history/memory is limited, favor items with broader category relevance.

            Output Requirements:
            - Rank ALL {len(candidate_items)} candidate items.
            - Return ONLY valid JSON.
            - Reasoning must be concise and avoid repeating item titles/categories.

            JSON Format:
            {{
            "ranked_item_ids": ["item_id1", "item_id2", "..."],
            "reasoning": "1 sentence explaining ranking logic, focusing on error avoidance and correction alignment"
            }}
            """
            else:
                prompt = f"""
            You are ranking candidate items for a user based on shopping history, item information, and fail-correction memories.
            {prompt_sample}
            Inputs:
            User Recent History (prioritize most recent):
            {json.dumps(user_profile[-10:], indent=2)}

            Collaborative Memory Insights (known wrong-choice patterns and correction rules):
            {json.dumps(memory_thoughts, indent=2)}

            Candidate Items:
            {json.dumps(candidate_info, indent=2)}

            Ranking Rules:
            - Prioritize candidates matching recent interaction history.
            - Use fail-correction memories to avoid known wrong-choice patterns.
            - Prefer candidates consistent with correction patterns from memory.
            - Ensure category consistency with history and correction signals.
            - If history/memory is limited, favor items with broader category relevance.

            Output Requirements:
            - Rank ALL {len(candidate_items)} candidate items.
            - Return ONLY valid JSON.
            - Reasoning must be concise and avoid repeating item titles/categories.

            JSON Format:
            {{
            "ranked_item_ids": ["item_id1", "item_id2", "..."],
            "reasoning": "1 sentence explaining ranking logic, focusing on history and correction alignment"
            }}
            """
        else:
            if eval_variant == "v2":
                prompt = f"""
            You are ranking candidate items for a user based on user memory, candidate item memory, and shopping history.
            {prompt_sample}
            Inputs:
            User Self Memory:
            {effective_user_memory}

            User Recent History (last 10 interactions, prioritize most recent):
            {json.dumps(user_profile[-10:], indent=2)}

            Candidate Items:
            {json.dumps(candidate_info, indent=2)}

            Ranking Rules:
            - Prioritize candidates matching user self memory and most recent interactions in history.
            - Use candidate item memory text to separate similar titles/categories.
            - Ensure strong category consistency with past purchases.
            - For similar items, prefer those with higher semantic similarity to recent history.
            - Avoid candidates that conflict with clear signals in recent preferred interactions.
            - If history is limited, favor items with broader category relevance.

            Output Requirements:
            - Rank ALL {len(candidate_items)} candidate items.
            - Return ONLY valid JSON.
            - Reasoning must be concise and avoid repeating item titles/categories.

            JSON Format:
            {{
            "ranked_item_ids": ["item_id1", "item_id2", "..."],
            "reasoning": "1 sentence explaining ranking logic, focusing on history alignment"
            }}
            """
            else:
                prompt = f"""
            You are ranking candidate items for a user based solely on their shopping history and item information.
            {prompt_sample}
            Inputs:
            User Recent History (last 10 interactions, prioritize most recent):
            {json.dumps(user_profile[-10:], indent=2)}

            Candidate Items:
            {json.dumps(candidate_info, indent=2)}

            Ranking Rules:
            - Prioritize candidates matching most recent interactions in history.
            - Ensure strong category consistency with past purchases.
            - For similar items, prefer those with higher semantic similarity to recent history.
            - Avoid candidates that conflict with clear signals in recent preferred interactions.
            - If history is limited, favor items with broader category relevance.

            Output Requirements:
            - Rank ALL {len(candidate_items)} candidate items.
            - Return ONLY valid JSON.
            - Reasoning must be concise and avoid repeating item titles/categories.

            JSON Format:
            {{
            "ranked_item_ids": ["item_id1", "item_id2", "..."],
            "reasoning": "1 sentence explaining ranking logic, focusing on history alignment"
            }}
            """

        try:
            response = self.qwen_generate(prompt=prompt, role_prompt="You are a ranker recommendation system.")
            result_text = response.strip()

            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            ranked_ids = result.get("ranked_item_ids", [])

            ranked_set = set(ranked_ids)
            all_candidate_ids = [c["item_id"] for c in candidate_items]
            missing = [cid for cid in all_candidate_ids if cid not in ranked_set]
            ranked_ids.extend(missing)
            return ranked_ids
        except Exception as e:
            print(f"Error in LLM ranking: {e}")
            return [c["item_id"] for c in candidate_items]

    def reset(self):
        self.behavior_memories = []
        self.user_interaction_history = []
        self.next_thought_id = 0

    def merge_from(self, other_system: "RecommendationMemorySystem"):
        offset = self.next_thought_id
        for mem in other_system.behavior_memories:
            mem.thought_id += offset
            mem.links = [link + offset for link in mem.links]
            self.behavior_memories.append(mem)
        self.next_thought_id += other_system.next_thought_id

    def get_evolution_statistics(self) -> Dict[str, Any]:
        if not self.behavior_memories:
            return {
                "total_memories": 0,
                "total_evolutions": 0,
                "avg_evolutions_per_memory": 0.0,
                "max_evolutions": 0,
                "min_evolutions": 0,
                "std_evolutions": 0.0,
                "memories_never_evolved": 0,
                "memories_evolved_once": 0,
                "memories_evolved_multiple": 0,
                "evolution_distribution": {},
                "top_10_most_evolved": [],
            }

        evolution_counts = [m.evolution_count for m in self.behavior_memories]
        stats = {
            "total_memories": len(self.behavior_memories),
            "total_evolutions": sum(evolution_counts),
            "avg_evolutions_per_memory": np.mean(evolution_counts),
            "max_evolutions": max(evolution_counts),
            "min_evolutions": min(evolution_counts),
            "std_evolutions": np.std(evolution_counts),
            "memories_never_evolved": sum(1 for c in evolution_counts if c == 0),
            "memories_evolved_once": sum(1 for c in evolution_counts if c == 1),
            "memories_evolved_multiple": sum(1 for c in evolution_counts if c > 1),
            "evolution_distribution": {
                f"{i}_times": sum(1 for c in evolution_counts if c == i) for i in range(max(evolution_counts) + 1)
            },
        }

        top_evolved = sorted(
            [(m.thought_id, m.evolution_count, m.behavior_explanation) for m in self.behavior_memories],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        stats["top_10_most_evolved"] = [
            {"thought_id": tid, "evolution_count": count, "behavior": behavior[:100]}
            for tid, count, behavior in top_evolved
        ]
        return stats

    def print_evolution_report(self):
        stats = self.get_evolution_statistics()

        print("\n" + "=" * 80)
        print("MEMORY EVOLUTION REPORT")
        print("=" * 80)
        if stats["total_memories"] == 0:
            print("No behavior memories available. Skipping evolution details.")
            return
        print(f"Total Memories: {stats['total_memories']}")
        print(f"Total Evolutions: {stats['total_evolutions']}")
        print(f"Average Evolutions per Memory: {stats['avg_evolutions_per_memory']:.2f}")
        print(f"Max Evolutions: {stats['max_evolutions']}")
        print(f"Min Evolutions: {stats['min_evolutions']}")
        print(f"Std Deviation: {stats['std_evolutions']:.2f}")
        print("-" * 80)
        print(f"Never Evolved: {stats['memories_never_evolved']}")
        print(f"Evolved Once: {stats['memories_evolved_once']}")
        print(f"Evolved Multiple Times: {stats['memories_evolved_multiple']}")
        print("-" * 80)
        print("Evolution Distribution:")
        for times, count in stats["evolution_distribution"].items():
            if count > 0:
                print(f"  {times}: {count} memories")
        print("-" * 80)
        print("Top 10 Most Evolved Memories:")
        for item in stats["top_10_most_evolved"]:
            print(f"  ID {item['thought_id']}: {item['evolution_count']} evolutions")
            print(f"    → {item['behavior']}")

    def save_memory(self, filepath: str, format: str = "json") -> None:
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        if format == "json":
            memories_dict = [mem.to_dict() for mem in self.behavior_memories]
            interactions_dict = [asdict(interaction) for interaction in self.user_interaction_history]
            data = {
                "behavior_memories": memories_dict,
                "user_interaction_history": interactions_dict,
                "next_thought_id": self.next_thought_id,
                "metadata": {
                    "num_memories": len(self.behavior_memories),
                    "num_interactions": len(self.user_interaction_history),
                    "save_timestamp": datetime.now().isoformat(),
                },
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ Memory saved to {filepath}")
            print("  - Format: JSON")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            print(f"  - File size: {file_size_mb:.2f} MB")
        elif format == "pickle":
            data = {
                "behavior_memories": self.behavior_memories,
                "user_interaction_history": self.user_interaction_history,
                "next_thought_id": self.next_thought_id,
                "metadata": {
                    "num_memories": len(self.behavior_memories),
                    "num_interactions": len(self.user_interaction_history),
                    "save_timestamp": datetime.now().isoformat(),
                },
            }
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ Memory saved to {filepath}")
            print("  - Format: Pickle")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            print(f"  - File size: {file_size_mb:.2f} MB")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")

    def load_memory(self, filepath: str, format: str = None) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        if format is None:
            if filepath.endswith(".json"):
                format = "json"
            elif filepath.endswith(".pkl") or filepath.endswith(".pickle"):
                format = "pickle"
            else:
                try:
                    with open(filepath, "r") as f:
                        json.load(f)
                    format = "json"
                except Exception:
                    format = "pickle"

        if format == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.behavior_memories = [BehaviorMemory.from_dict(mem_dict) for mem_dict in data["behavior_memories"]]
            self.user_interaction_history = [
                UserInteraction(**interaction_dict) for interaction_dict in data["user_interaction_history"]
            ]
            self.next_thought_id = data["next_thought_id"]
            print(f"✓ Memory loaded from {filepath}")
            print("  - Format: JSON")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
        elif format == "pickle":
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.behavior_memories = data["behavior_memories"]
            self.user_interaction_history = data["user_interaction_history"]
            self.next_thought_id = data["next_thought_id"]
            print(f"✓ Memory loaded from {filepath}")
            print("  - Format: Pickle")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")
