import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
# import google.generativeai as genai
from collections import defaultdict
from tqdm import tqdm
import random
import pickle
import time
import re
import urllib.request
import urllib.error
import argparse
from pathlib import Path

try:
    from utils import set_seed
except ImportError:
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
# # api_key = 'AIzaSyDNFlv44-Gl6QWVKFdwXEtOrlRtv4bBTu8'
# api_key = 'AIzaSyD98UePYe0I55FTfN9CjA3uShBhw9K02DY'

# genai.configure(api_key=api_key)
# print("✓ Gemini API configured: {}".format(api_key) )
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# np.random.seed(42)
# random.seed(42)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
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

@dataclass
class UserInteraction:
    """Represents a single user-item interaction"""
    item_id: str
    item_name: str
    item_category: str
    action_type: str  # 'purchase' for implicit feedback
    rating: Optional[float] = None
    timestamp: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class BehaviorMemory:
    """
    Represents a generalized thought about user behavior patterns
    """
    thought_id: int
    interaction_sequence: List[UserInteraction]
    behavior_explanation: str
    pattern_description: str
    # extracted_preferences: List[str]
    keywords: List[str]
    embedding: np.ndarray
    links: List[int] = field(default_factory=list)
    timestamp: str = None
    evolution_count: int = 0  # Số lần đã evolve
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)  # Lịch sử evolution
    max_evolutions: Optional[int] = None  # Giới hạn số lần evolve (None = unlimited)
    last_evolved_timestamp: Optional[str] = None  # Lần evolve cuối

    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    def can_evolve(self) -> bool:
        """Kiểm tra xem memory này còn được phép evolve không"""
        if self.max_evolutions is None:
            return True
        return self.evolution_count < self.max_evolutions
    def record_evolution(self, 
                        update_type: str,
                        old_values: Dict[str, Any],
                        new_values: Dict[str, Any],
                        reasoning: str) -> None:
        """Ghi lại một lần evolution"""
        self.evolution_count += 1
        self.last_evolved_timestamp = datetime.now().isoformat()
        
        self.evolution_history.append({
            'evolution_number': self.evolution_count,
            'timestamp': self.last_evolved_timestamp,
            'update_type': update_type,
            'old_values': old_values,
            'new_values': new_values,
            'reasoning': reasoning
        })
    def to_dict(self):
        data = asdict(self)
        data['embedding'] = self.embedding.tolist()
        data['interaction_sequence'] = [asdict(i) for i in self.interaction_sequence]
        return data
    
    @classmethod
    def from_dict(cls, data):
        data['embedding'] = np.array(data['embedding'])
        data['interaction_sequence'] = [UserInteraction(**i) for i in data['interaction_sequence']]
        return cls(**data)


@dataclass
class AgentCFUserState:
    """AgentCF-style user state used to bootstrap fail-interaction memory generation."""
    user_id: str
    short_term_memory: str = "I enjoy discovering new items."
    long_term_memory: List[str] = field(default_factory=list)
    interaction_history: List[str] = field(default_factory=list)

    def update_memory(self, new_memory: str):
        self.long_term_memory.append(self.short_term_memory)
        self.short_term_memory = new_memory

    def add_interaction(self, item_id: str):
        self.interaction_history.append(item_id)


@dataclass
class AgentCFItemState:
    """AgentCF-style item state with mutable textual memory."""
    item_id: str
    title: str
    category: str
    memory: str

class RecommendationMemorySystem:
    """A-Mem adapted for Amazon product recommendation with Memory Evolution"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_gemini_embeddings: bool = None,
                 chat_api_base: Optional[str] = None,
                 embedding_api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 chat_model_name: Optional[str] = None,
                 embedding_model_name: Optional[str] = None):
        _ = use_gemini_embeddings  # kept for backward compatibility

        self.llm_name = model_name
        self.embedding_model_name = embedding_model_name or os.getenv("embedding_model_name") or embedding_model
        self.chat_model_name = chat_model_name or os.getenv("chat_model_name") or model_name
        self.chat_api_base = (chat_api_base or os.getenv("chat_api_base") or os.getenv("api_base") or "").rstrip("/")
        self.embedding_api_base = (embedding_api_base or os.getenv("embedding_api_base") or os.getenv("api_base") or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"

        self.use_api_chat = bool(self.chat_api_base)
        # VIRAL-style embedding path: always use local SentenceTransformer.
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
                device_map="auto"
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

    def qwen_generate(self, prompt: str, role_prompt = "You are a helpful AI assistant.", max_new_tokens=8000) -> str:
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
                return "".join(
                    chunk.get("text", "") for chunk in content if isinstance(chunk, dict)
                )
            return str(content)

        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)


    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def _create_embedding(self, text: str) -> np.ndarray:
        emb = self.embedding_model.encode(
            str(text),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)

    def _simple_hash_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def add_interaction(self, 
                       item_id: str,
                       item_name: str,
                       item_category: str,
                       action_type: str = "purchase",
                       rating: Optional[float] = None,
                       metadata: Optional[Dict] = None) -> UserInteraction:
        interaction = UserInteraction(
            item_id=item_id,
            item_name=item_name,
            item_category=item_category,
            action_type=action_type,
            rating=rating,
            metadata=metadata or {}
        )
        
        self.user_interaction_history.append(interaction)
        
        return interaction
    
    def create_behavior_thought(self, 
                               interaction_window: List[UserInteraction],
                               k_neighbors: int = 10) -> BehaviorMemory:
        interaction_summary = []
        for interaction in interaction_window:
            summary = {
                "item": interaction.item_name,
                "category": interaction.item_category,
                "action": interaction.action_type
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
            # response = self.model.generate_content(prompt)
            response = self.qwen_generate(prompt=prompt, role_prompt='You are a behavioral memory modeling system.')
            # time.sleep(5)
            result_text = response.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            behavior_explanation = result.get("behavior_explanation", "")
            pattern_description = result.get("pattern_description", "")
            keywords = result.get("keywords", [])
            # extracted_preferences = result.get("extracted_preferences", [])
            
        except Exception as e:
            print(f"Error in behavior analysis: {e}")
            behavior_explanation = f"A failed interaction occurred with {len(interaction_window)} compared items."
            pattern_description = "Correction rule is unclear; prefer signals from the preferred item over the wrong choice."
            # extracted_preferences = []
            keywords = [i.item_category for i in interaction_window[:3]]
        
        combined_text = f"{behavior_explanation} {pattern_description} {' '.join(keywords)}"
        embedding = self._create_embedding(combined_text)
        
        behavior_memory = BehaviorMemory(
            thought_id=self.next_thought_id,
            interaction_sequence=interaction_window.copy(),
            behavior_explanation=behavior_explanation,
            pattern_description=pattern_description,
            # extracted_preferences=extracted_preferences,
            keywords=keywords,
            embedding=embedding
        )
        
        self.next_thought_id += 1
        return behavior_memory
    
    def link_behavior_memories(self, 
                               new_memory: BehaviorMemory,
                               k: int = 5, wo_link=False) -> List[int]:
        """Link new behavior memory with similar past patterns"""
        if len(self.behavior_memories) == 0:
            return []
        
        similarities = []
        for memory in self.behavior_memories:
            sim = self._cosine_similarity(new_memory.embedding, memory.embedding)
            similarities.append((memory.thought_id, sim, memory))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_k = similarities[:min(k, len(similarities))]
        
        if len(nearest_k) == 0:
            return []
        if wo_link:
            return [thought_id for thought_id, _, _ in nearest_k]
        
        nearest_info = []
        for thought_id, sim, memory in nearest_k:
            nearest_info.append({
                "thought_id": thought_id,
                "behavior_explanation": memory.behavior_explanation,
                "pattern": memory.pattern_description,
                # "preferences": memory.extracted_preferences,
                "similarity": float(sim)
            })
        
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
            # response = self.model.generate_content(prompt)
            # time.sleep(3)
            response = self.qwen_generate(prompt=prompt, role_prompt='You are a behavioral memory modeling system.')

            result_text = response.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            if result.get("should_link", False):
                return result.get("linked_thought_ids", [])
            else:
                return []
                
        except Exception as e:
            print(f"Error in linking: {e}")
            return [thought_id for thought_id, sim, _ in nearest_k if sim > 0.65]
    
    def evolve_behavior_memories(self,
                                new_memory: BehaviorMemory,
                                linked_ids: List[int],
                                max_evolutions_per_memory: Optional[int] = None) -> None:
        """Evolve existing behavior memories based on new patterns (Section 3.3)"""
        if len(linked_ids) == 0:
            return

        linked_memories = [m for m in self.behavior_memories if m.thought_id in linked_ids]
        if len(linked_memories) == 0:
            return
        # ============ LỌC MEMORIES CÒN CÓ THỂ EVOLVE ============
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
            mem_info.append({
                "thought_id": mem.thought_id,
                "behavior_explanation": mem.behavior_explanation,
                "pattern": mem.pattern_description,
                "evolution_count": mem.evolution_count 
            })
        

# Return ONLY JSON."""
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
            # response = self.model.generate_content(prompt)
            # time.sleep(3)
            response = self.qwen_generate(prompt=prompt, role_prompt='You are a behavioral memory modeling system.')
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
                        # ============ GHI LẠI GIÁ TRỊ CŨ ============
                        old_values = {
                            'behavior_explanation': memory.behavior_explanation,
                            'pattern_description': memory.pattern_description,
                            # 'extracted_preferences': memory.extracted_preferences.copy()
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
                        
                        # if update.get("additional_preferences"):
                        #     memory.extracted_preferences.extend(update["additional_preferences"])
                        #     memory.extracted_preferences = list(set(memory.extracted_preferences))
                        #     updated = True
                        #     update_type.append("preferences")
                        
                        # Regenerate embedding if updated
                        if updated:
                            # combined_text = f"{memory.behavior_explanation} {memory.pattern_description} {' '.join(memory.keywords)} {' '.join(memory.extracted_preferences)}"
                            combined_text = f"{memory.behavior_explanation} {memory.pattern_description} {' '.join(memory.keywords)}"
                            memory.embedding = self._create_embedding(combined_text)
                            new_values = {
                                'behavior_explanation': memory.behavior_explanation,
                                'pattern_description': memory.pattern_description,
                                # 'extracted_preferences': memory.extracted_preferences.copy()
                            }
                            
                            memory.record_evolution(
                                update_type=", ".join(update_type),
                                old_values=old_values,
                                new_values=new_values,
                                reasoning=update.get("reasoning", "")
                            )
                        
        except Exception as e:
            print(f"Error in memory evolution: {e}")
    
    def add_behavior_memory(self,
                           interaction_window: List[UserInteraction],
                           k_neighbors: int = 5) -> BehaviorMemory:
        """Complete A-Mem pipeline: Create, Link, and Evolve"""
        # Step 1: Create behavior thought
        behavior_memory = self.create_behavior_thought(interaction_window, k_neighbors)
        
        # Step 2: Link with similar patterns
        linked_ids = self.link_behavior_memories(behavior_memory, k_neighbors)
        behavior_memory.links = linked_ids
        
        # Update bidirectional links
        for thought_id in linked_ids:
            memory = next((m for m in self.behavior_memories if m.thought_id == thought_id), None)
            if memory and behavior_memory.thought_id not in memory.links:
                memory.links.append(behavior_memory.thought_id)
        
        # Step 3: Evolve existing memories based on new pattern
        self.evolve_behavior_memories(behavior_memory, linked_ids)
        
        # Add to collection
        self.behavior_memories.append(behavior_memory)
        return behavior_memory
    
    def retrieve_relevant_memories(self, user_profile_text: str, k: int = 5) -> List[BehaviorMemory]:
        """Retrieve top-k most relevant memories based on user profile"""
        if len(self.behavior_memories) == 0:
            return []
        
        # Create embedding from user profile
        profile_embedding = self._create_embedding(user_profile_text)
        
        # Calculate similarities with all memories
        similarities = []
        for memory in self.behavior_memories:
            sim = self._cosine_similarity(profile_embedding, memory.embedding)
            similarities.append((memory, sim))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_memories = [mem for mem, sim in similarities[:k]]
        
        return top_memories
    
    def llm_ranking(self, 
                   train_items: List[Dict],
                   candidate_items: List[Dict],
                   retrieved_memories: List[BehaviorMemory],
                   prompt_sample: str,
                   user_memory: Optional[str] = None,
                   eval_variant: str = "v2") -> List[str]:
        """Use LLM to rank candidate items based on user profile, item memories, and behavior memories."""
        
        # Prepare user profile from train items
        user_profile = []
        for item in train_items:
            user_profile.append({
                "title": item['title'],
                "category": item['category']
            })
        # Prepare candidate items
        candidate_info = []
        for item in candidate_items:
            if eval_variant == "v2":
                candidate_info.append({
                    "item_id": item['item_id'],
                    "title": item['title'],
                    "category": item['category'],
                    "memory": item.get('memory', '')
                })
            else:
                candidate_info.append({
                    "item_id": item['item_id'],
                    "title": item['title'],
                    "category": item['category'],
                })
        effective_user_memory = user_memory or "I enjoy discovering new items."
        
        # Prepare memory thoughts
        if retrieved_memories is not None:
            memory_thoughts = []
            for mem in retrieved_memories:
                memory_thoughts.append({
                    "behavior_explanation": mem.behavior_explanation,
                    "pattern": mem.pattern_description,
                    # "preferences": mem.extracted_preferences
                })
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
            # prompt = f"""
            #         You are ranking candidate items for a user.

            # Inputs:

            # User Item History (recent interactions only):
            # {json.dumps(user_profile[-10:], indent=2)}

            # User Collaborative Memory (high-level behavior insights):
            # {json.dumps(memory_thoughts, indent=2)}

            # Candidate Items:
            # {json.dumps(candidate_info, indent=2)}

            # Ranking rules:
            # - Base the ranking ONLY on the provided history and collaborative memory.
            # - Prefer category and intent consistency.
            # - If items are similar, rank the more generally relevant one higher.

            # Output requirements:
            # - Rank ALL {len(candidate_items)} candidate items.
            # - Return ONLY valid JSON.
            # - Do NOT include item titles in the reasoning.

            # JSON format:
            # {{
            # "ranked_item_ids": ["item_id1", "item_id2", "..."],
            # "reasoning": "one concise sentence explaining the overall ranking criteria (1-2 sentences)"
            # }}

            #         """
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
            # prompt = f"""
            # You are ranking candidate items for a user.

            # Inputs:

            # User Item History (recent interactions only):
            # {json.dumps(user_profile[-10:], indent=2)}

            # Candidate Items:
            # {json.dumps(candidate_info, indent=2)}

            # Ranking rules:
            # - Base the ranking ONLY on the provided user item history.
            # - Prioritize category consistency and semantic similarity to past items.
            # - If multiple items are similar, rank the more generally relevant one higher.

            # Output requirements:
            # - Rank ALL {len(candidate_items)} candidate items.
            # - Return ONLY valid JSON.
            # - Do NOT include item titles or categories in the reasoning.

            # JSON format:
            # {{
            # "ranked_item_ids": ["item_id1", "item_id2", "..."],
            # "reasoning": "one concise sentence explaining the overall ranking logic"
            # }}
            # """

        try:
            # print("LLM Ranking Prompt:\n", prompt)
            # response = self.model.generate_content(prompt)
            # time.sleep(3)
            response = self.qwen_generate(prompt=prompt, role_prompt='You are a ranker recommendation system.')
            result_text = response.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            ranked_ids = result.get("ranked_item_ids", [])
            
            # Ensure all candidates are in the ranking
            ranked_set = set(ranked_ids)
            all_candidate_ids = [c['item_id'] for c in candidate_items]
            missing = [cid for cid in all_candidate_ids if cid not in ranked_set]
            ranked_ids.extend(missing)
            
            return ranked_ids
            
        except Exception as e:
            print(f"Error in LLM ranking: {e}")
            # Fallback: return in original order
            return [c['item_id'] for c in candidate_items]

   

    def reset(self):
        """Reset the memory system for a new user"""
        self.behavior_memories = []
        self.user_interaction_history = []
        self.next_thought_id = 0
    
    def merge_from(self, other_system: 'RecommendationMemorySystem'):
        """Merge memories from another system into this one"""
        offset = self.next_thought_id
        for mem in other_system.behavior_memories:
            mem.thought_id += offset
            mem.links = [link + offset for link in mem.links]
            self.behavior_memories.append(mem)
        self.next_thought_id += other_system.next_thought_id
        # Optionally merge histories if needed, but since per-user, maybe not
        # self.user_interaction_history.extend(other_system.user_interaction_history)
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Phân tích thống kê về evolution của các memories"""
        if not self.behavior_memories:
            return {
                'total_memories': 0,
                'total_evolutions': 0,
                'avg_evolutions_per_memory': 0.0,
                'max_evolutions': 0,
                'min_evolutions': 0,
                'std_evolutions': 0.0,
                'memories_never_evolved': 0,
                'memories_evolved_once': 0,
                'memories_evolved_multiple': 0,
                'evolution_distribution': {},
                'top_10_most_evolved': [],
            }
        
        evolution_counts = [m.evolution_count for m in self.behavior_memories]
        
        stats = {
            'total_memories': len(self.behavior_memories),
            'total_evolutions': sum(evolution_counts),
            'avg_evolutions_per_memory': np.mean(evolution_counts),
            'max_evolutions': max(evolution_counts),
            'min_evolutions': min(evolution_counts),
            'std_evolutions': np.std(evolution_counts),
            'memories_never_evolved': sum(1 for c in evolution_counts if c == 0),
            'memories_evolved_once': sum(1 for c in evolution_counts if c == 1),
            'memories_evolved_multiple': sum(1 for c in evolution_counts if c > 1),
            'evolution_distribution': {
                f'{i}_times': sum(1 for c in evolution_counts if c == i)
                for i in range(max(evolution_counts) + 1)
            }
        }
        
        # Top memories theo evolution count
        top_evolved = sorted(
            [(m.thought_id, m.evolution_count, m.behavior_explanation) 
            for m in self.behavior_memories],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        stats['top_10_most_evolved'] = [
            {
                'thought_id': tid,
                'evolution_count': count,
                'behavior': behavior[:100]  # Truncate
            }
            for tid, count, behavior in top_evolved
        ]
        
        return stats

    def print_evolution_report(self):
        """In báo cáo evolution"""
        stats = self.get_evolution_statistics()
        
        print("\n" + "="*80)
        print("MEMORY EVOLUTION REPORT")
        print("="*80)
        if stats['total_memories'] == 0:
            print("No behavior memories available. Skipping evolution details.")
            return
        print(f"Total Memories: {stats['total_memories']}")
        print(f"Total Evolutions: {stats['total_evolutions']}")
        print(f"Average Evolutions per Memory: {stats['avg_evolutions_per_memory']:.2f}")
        print(f"Max Evolutions: {stats['max_evolutions']}")
        print(f"Min Evolutions: {stats['min_evolutions']}")
        print(f"Std Deviation: {stats['std_evolutions']:.2f}")
        print("-"*80)
        print(f"Never Evolved: {stats['memories_never_evolved']}")
        print(f"Evolved Once: {stats['memories_evolved_once']}")
        print(f"Evolved Multiple Times: {stats['memories_evolved_multiple']}")
        print("-"*80)
        print("Evolution Distribution:")
        for times, count in stats['evolution_distribution'].items():
            if count > 0:
                print(f"  {times}: {count} memories")
        print("-"*80)
        print("Top 10 Most Evolved Memories:")
        for item in stats['top_10_most_evolved']:
            print(f"  ID {item['thought_id']}: {item['evolution_count']} evolutions")
            print(f"    → {item['behavior']}")

    def save_memory(self, filepath: str, format: str = 'json') -> None:
        """
        Lưu memory system ra file (chứa memories của TẤT CẢ users)
        
        Args:
            filepath: Đường dẫn file để lưu
            format: Định dạng file ('json' hoặc 'pickle')
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        if format == 'json':
            memories_dict = [mem.to_dict() for mem in self.behavior_memories]
            interactions_dict = [asdict(interaction) for interaction in self.user_interaction_history]
            
            data = {
                'behavior_memories': memories_dict,
                'user_interaction_history': interactions_dict,
                'next_thought_id': self.next_thought_id,
                'metadata': {
                    'num_memories': len(self.behavior_memories),
                    'num_interactions': len(self.user_interaction_history),
                    'save_timestamp': datetime.now().isoformat()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            file_size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✓ Memory saved to {filepath}")
            print(f"  - Format: JSON")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            print(f"  - File size: {file_size_mb:.2f} MB")
            
        elif format == 'pickle':
            data = {
                'behavior_memories': self.behavior_memories,
                'user_interaction_history': self.user_interaction_history,
                'next_thought_id': self.next_thought_id,
                'metadata': {
                    'num_memories': len(self.behavior_memories),
                    'num_interactions': len(self.user_interaction_history),
                    'save_timestamp': datetime.now().isoformat()
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            file_size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✓ Memory saved to {filepath}")
            print(f"  - Format: Pickle")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            print(f"  - File size: {file_size_mb:.2f} MB")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")

    def load_memory(self, filepath: str, format: str = None) -> None:
        """
        Tải memory system từ file
        
        Args:
            filepath: Đường dẫn file để đọc
            format: Định dạng file ('json' hoặc 'pickle'). Nếu None, tự động detect từ extension
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format is None:
            if filepath.endswith('.json'):
                format = 'json'
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'
            else:
                try:
                    with open(filepath, 'r') as f:
                        json.load(f)
                    format = 'json'
                except:
                    format = 'pickle'
        
        if format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.behavior_memories = [
                BehaviorMemory.from_dict(mem_dict) 
                for mem_dict in data['behavior_memories']
            ]
            
            self.user_interaction_history = [
                UserInteraction(**interaction_dict)
                for interaction_dict in data['user_interaction_history']
            ]
            
            self.next_thought_id = data['next_thought_id']
            
            print(f"✓ Memory loaded from {filepath}")
            print(f"  - Format: JSON")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.behavior_memories = data['behavior_memories']
            self.user_interaction_history = data['user_interaction_history']
            self.next_thought_id = data['next_thought_id']
            
            print(f"✓ Memory loaded from {filepath}")
            print(f"  - Format: Pickle")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")

import os
from typing import Dict, List

def save_all_users_ranking_results(all_results: List[Dict],
                                  items_meta: Dict,
                                  output_file: str = "all_users_ranking_results.json"):
    """
    Lưu toàn bộ kết quả ranking của tất cả users vào 1 file JSON duy nhất.
    
    Args:
        all_results: List các dict chứa thông tin của từng user
        items_meta: Metadata items để lấy title, category,...
        output_file: Tên file output (sẽ tự tạo thư mục nếu cần)
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    def get_item_info(item_id: str) -> Dict:
        if item_id in items_meta:
            info = items_meta[item_id]
            return {
                "item_id": item_id,
                "title": info.get("title", ""),
                "category": info.get("main_cat", "Unknown"),
                # "brand": info.get("brand", ""),
                # "price": info.get("price", None)
            }
        else:
            return {
                "item_id": item_id,
                "title": f"Unknown Item {item_id}",
                "category": "Unknown",
                # "brand": "",
                # "price": None
            }
    
    # Chuyển đổi chi tiết items cho tất cả users
    final_results = []
    for res in all_results:
        user_result = {
            "user_id": res["user_id"],
            "num_candidates": len(res["candidates"]),
            "ground_truth_item_ids": res["ground_truth"],
            "candidate_item_ids": res["candidates"],
            "reranked_item_ids": res["predictions"],
            # "ground_truth_items": [get_item_info(iid) for iid in res["ground_truth"]],
            "candidate_items": [get_item_info(iid) for iid in res["candidates"]],
            "reranked_items": [get_item_info(iid) for iid in res["predictions"]],
            "metrics": res["metrics"],  # thêm metrics của user này
            "baseline_metrics": res["baseline_metrics"]
        }
        final_results.append(user_result)
    
    # Lưu vào 1 file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved ranking results of {len(final_results)} users to {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def load_data(items_path: str, sequences_path: str, negatives_path: str):
    """Load Amazon dataset"""
    print("Loading data...")
    
    with open(items_path, 'r', encoding="utf-8") as f:
        items_meta = json.load(f)
    
    with open(sequences_path, 'r', encoding="utf-8") as f:
        user_sequences = json.load(f)
    
    with open(negatives_path, 'r', encoding="utf-8") as f:
        user_negatives = json.load(f)
    
    print(f"Loaded {len(items_meta)} items")
    print(f"Loaded {len(user_sequences)} users")
    
    return items_meta, user_sequences, user_negatives

def calculate_recall_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate Recall@K"""
    top_k = predictions[:k]
    hits = len(set(top_k) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0

def calculate_ndcg_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate NDCG@K"""
    top_k = predictions[:k]
    
    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    
    # IDCG
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])
    
    return dcg / idcg if idcg > 0 else 0.0


def init_agentcf_item_states(items_meta: Dict[str, Dict[str, Any]]) -> Dict[str, AgentCFItemState]:
    """Initialize item states exactly like AgentCF item agent initialization."""
    item_states: Dict[str, AgentCFItemState] = {}
    for item_id, item_info in items_meta.items():
        title = item_info.get("title", f"Item {item_id}")
        category = item_info.get("main_cat", "Unknown")
        memory = f"The item is called '{title}'. The category is: '{category}'."
        item_states[item_id] = AgentCFItemState(
            item_id=item_id,
            title=title,
            category=category,
            memory=memory,
        )
    return item_states


def get_or_create_user_state(user_states: Dict[str, AgentCFUserState], user_id: str) -> AgentCFUserState:
    if user_id not in user_states:
        user_states[user_id] = AgentCFUserState(user_id=user_id)
    return user_states[user_id]


def save_agent_states(
    user_states: Dict[str, AgentCFUserState],
    item_states: Dict[str, AgentCFItemState],
    filepath: str,
) -> None:
    """Persist updated user/item states so evaluation can always reuse latest memories."""
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
    """Load persisted user/item states with updated memories."""
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


def _extract_json_from_llm_output(raw_output: str) -> Dict[str, Any]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).rstrip("```").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group(0))


def autonomous_interaction_agentcf(
    memory_system: RecommendationMemorySystem,
    user_state: AgentCFUserState,
    pos_item: AgentCFItemState,
    neg_item: AgentCFItemState,
) -> Tuple[str, str]:
    """AgentCF-style autonomous interaction: choose between positive/negative item."""
    prompt = f"""You are an enthusiast. Here is your self-introduction: "{user_state.short_term_memory}"

Now, you are considering to select an item from two candidates:
1. Title: {neg_item.title}, Description: {neg_item.memory}
2. Title: {pos_item.title}, Description: {pos_item.memory}
\n\n Please select the item that aligns best with your preferences and explain your choice while rejecting the other. \n Follow these steps:\n 1. Extract your preferences and dislikes from your self-introduction. \n 2. Evaluate the two items based on your preferences and how they relate to the item features.\n 3. Explain your choice, detailing the relationship between your preferences/dislikes and the item features

\n\n Important notes:
\n 1. Do not fabricate your preferences! If your self-introduction lacks relevant details, use common knowledge to guide your decision, such as item popularity. \n 2. Select one candidate, not both. \n 3. Your explanation should be specific; general preferences like genre are insufficient. Focus on the item's finer attributes and be concise! \n 4. Base your explanation on facts. If your self-introduction doesn't specify preferences, you cannot claim your decision was influenced by them."

Output format:
Chosen Item: [1 or 2]
Explanation: [Your detailed reasoning]

Important: You must choose one of these two candidates."""

    response = memory_system.qwen_generate(prompt=prompt)
    chosen_item_id = pos_item.item_id
    if "Chosen Item: 1" in response or "chosen item: 1" in response.lower():
        chosen_item_id = neg_item.item_id
    return chosen_item_id, response


def collaborative_reflection_agentcf(
    memory_system: RecommendationMemorySystem,
    user_state: AgentCFUserState,
    pos_item: AgentCFItemState,
    neg_item: AgentCFItemState,
    chosen_item_id: str,
    explanation: str,
) -> None:
    """AgentCF-style reflection update for user memory and item memories."""
    if chosen_item_id == pos_item.item_id:
        return

    user_prompt = f"""You are an enthusiast with these preferences: "{user_state.short_term_memory}"

Recently, you chose between two items:
1. Title: {neg_item.title}, Description: {neg_item.memory}
2. Title: {pos_item.title}, Description: {pos_item.memory}

You selected item 1, but you discovered you actually prefer item 2 instead.
Your previous explanation was: "{explanation}"

This indicates an incorrect choice, and your previous judgment about your preferences was mistaken. Your task now is to update your self-introduction with your new preferences and dislikes. \n Follow these steps: \n 1. Analyze misconceptions in your previous judgment and correct them.\n 2. Identify new preferences from '{pos_item.title}' and dislikes from '{neg_item.title}'. \n 3. Summarize your past preferences, merging them with new insights and removing conflicting parts.\n 4. Update your self-introduction, starting with new preferences, then summarizing past ones, followed by dislikes. \n\n Important notes: 1. Keep it under 150 words.  \n 2. Be concise and clear. \n 3. Describe only the features of items you prefer or dislike, without mentioning your thought process. \n 4. Your self-introduction should be specific and personalized; avoid generic preferences."

Output format:
My updated self-introduction: [Your updated preferences in under 150 words]

Important: Focus on what features you like and dislike, be specific and personalized."""

    new_user_memory = memory_system.qwen_generate(prompt=user_prompt)
    if "My updated self-introduction:" in new_user_memory:
        new_user_memory = new_user_memory.split("My updated self-introduction:")[1].strip()
    user_state.update_memory(new_user_memory)

    item_prompt = f"""A user with these preferences browsed items: "{user_state.short_term_memory}"

The user considered two items:
1. Title: {pos_item.title}, Description: {pos_item.memory}
2. Title: {neg_item.title}, Description: {neg_item.memory}

The user initially chose item 2 but actually prefers item 1, indicating the descriptions may be misleading.

Your task is to update the descriptions of these items based on these insights. \n Follow these steps:\n 1. Analyze the user's preferences and dislikes from the self-description. \n 2. Explore the chosen item's features that align with preferences and oppose dislikes, and examine the rejected item's features that align with dislikes and oppose preferences. Highlight the differences thoroughly. \n 3. Incorporate new features into the previous descriptions, preserving key information while being concise.\n\n Important notes: \n 1. Your output should be in the following format: 'The updated description of the first item is: [updated description]. \\n The updated description of the second item is: [updated description].'. \n 2. Each updated description cannot exceed 50 words; be concise and clear! \n 3. In your updated descriptions, refer to preferences collectively, avoiding individual references. For example, say 'the user with ... preferences/dislikes'.\n 4. New features should reflect user preferences, and the updated descriptions must not contradict the inherent characteristics of the items, e.g., do not describe a thriller as having a predictably happy ending.

Update the description of item 1 to better reflect why users with these preferences would like it.

Output format (STRICT JSON, no extra text):
{{
  "item_1": "<updated description, single paragraph>",
  "item_2": "<updated description, single paragraph>"
}}

Important: Make it specific and aligned with user preferences."""

    new_item_memory = memory_system.qwen_generate(prompt=item_prompt)

    raw_output = new_item_memory.strip()

    # 1. Remove markdown code fences if any
    if raw_output.startswith("```"):
        raw_output = re.sub(r"^```[a-zA-Z]*", "", raw_output)
        raw_output = raw_output.rstrip("```").strip()

    # 2. Extract JSON object safely
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")

    json_str = match.group(0)

    # 3. Load JSON
    try:
        data = json.loads(json_str)
        item1_desc = data["item_1"].strip()
        item2_desc = data["item_2"].strip()
    except Exception as e:
        raise ValueError(f"Invalid JSON output from LLM: {e}")

    # 5. Update memories
    pos_item.memory = item1_desc
    neg_item.memory = item2_desc


def train_memory_from_fail_interactions(
    user_id: str,
    user_data: Dict,
    memory_system: RecommendationMemorySystem,
    user_states: Dict[str, AgentCFUserState],
    item_states: Dict[str, AgentCFItemState],
    max_iterations: int = 1,
) -> List[BehaviorMemory]:
    """
    Hybrid training:
    - AgentCF-style initialization and interaction loop.
    - Create behavior memories ONLY from failed interactions.
    """
    train_items = user_data["train"][-30:]
    if len(train_items) == 0:
        return []

    user_state = get_or_create_user_state(user_states, user_id)
    all_item_ids = list(item_states.keys())
    if not all_item_ids:
        return []
    blocked_items: Set[str] = set(train_items) | set(user_state.interaction_history)

    def sample_negative_item_fast(pos_item_id: str, max_tries: int = 64) -> Optional[str]:
        """
        Fast negative sampling:
        - sample directly from full catalog
        - reject if candidate is positive item or inside user's blocked/history items
        """
        total_items = len(all_item_ids)
        if total_items <= 1:
            return None

        for _ in range(max_tries):
            cand = random.choice(all_item_ids)
            if cand != pos_item_id and cand not in blocked_items:
                return cand

        # Rare fallback to guarantee a valid candidate if one exists.
        start = random.randrange(total_items)
        for offset in range(total_items):
            cand = all_item_ids[(start + offset) % total_items]
            if cand != pos_item_id and cand not in blocked_items:
                return cand
        return None

    # temp memory system must be local for this user
    memory_system.user_interaction_history = []
    memory_system.behavior_memories = []
    memory_system.next_thought_id = 0

    new_memories: List[BehaviorMemory] = []
    for pos_item_id in train_items:
        if pos_item_id not in item_states:
            continue

        neg_item_id = sample_negative_item_fast(pos_item_id)
        if neg_item_id is None:
            continue

        pos_item = item_states[pos_item_id]
        neg_item = item_states[neg_item_id]

        for _ in range(max_iterations):
            chosen_item_id, explanation = autonomous_interaction_agentcf(
                memory_system=memory_system,
                user_state=user_state,
                pos_item=pos_item,
                neg_item=neg_item,
            )

            if chosen_item_id == pos_item_id:
                user_state.add_interaction(pos_item_id)
                break

            try:
                collaborative_reflection_agentcf(
                    memory_system=memory_system,
                    user_state=user_state,
                    pos_item=pos_item,
                    neg_item=neg_item,
                    chosen_item_id=chosen_item_id,
                    explanation=explanation,
                )
            except Exception as e:
                print(f"  ⚠ Reflection failed for user {user_id}, item {pos_item_id}: {e}")
                continue

            # Memory unit is one failed interaction pair instead of sliding windows.
            fail_window = [
                UserInteraction(
                    item_id=neg_item.item_id,
                    item_name=neg_item.title,
                    item_category=neg_item.category,
                    action_type="wrong_choice",
                    metadata={"user_id": user_id, "role": "chosen_wrong"},
                ),
                UserInteraction(
                    item_id=pos_item.item_id,
                    item_name=pos_item.title,
                    item_category=pos_item.category,
                    action_type="preferred_item",
                    metadata={"user_id": user_id, "role": "ground_truth"},
                ),
            ]
            try:
                fail_memory = memory_system.create_behavior_thought(fail_window)
                new_memories.append(fail_memory)
            except Exception as e:
                print(f"  ⚠ Fail-memory creation error for user {user_id}: {e}")

    return new_memories

def evaluate_user(user_data: Dict,
                 negative_data: Dict,
                 items_meta: Dict,
                 memory_system: RecommendationMemorySystem,
                 eval_type: str = 'test',
                 use_memory = True,
                 k_memories: int = 5,
                 sample_user_list: List = None,
                 negative_data_sample_list: List = None,
                 user_state: Optional[AgentCFUserState] = None,
                 item_states: Optional[Dict[str, AgentCFItemState]] = None,
                 eval_variant: str = "v2",
                 fixed_candidates: Optional[List[str]] = None) -> Dict[str, float]:
    """Evaluate for a single user with LLM-based ranking"""
    
    # Get ground truth and candidates
    if eval_type == 'val':
        ground_truth = user_data['val']
        negatives = negative_data.get('val_neg', [])
    else:  # test
        ground_truth = user_data['test']
        negatives = negative_data.get('test_neg', [])

    if sample_user_list is not None:
        ground_truth_sample_fewshot = []
        negatives_sample_fewshot = []
        for i in range(len(sample_user_list)):
            sample_user_data = sample_user_list[i]
            negative_data_sample = negative_data_sample_list[i]

            ground_truth_sample = sample_user_data.get('val', [])
            ground_truth_sample_fewshot.append(ground_truth_sample)

            negatives_sample = negative_data_sample.get('val_neg', [])
            negatives_sample_fewshot.append(negatives_sample)

    # Prepare train items for user profile
    train_items_info = []
    user_profile_texts = []
    for item_id in user_data['train'][-10:]:
        item_state = item_states.get(item_id) if item_states else None
        if item_state is not None:
            title = item_state.title
            category = item_state.category
        elif item_id in items_meta:
            item_info = items_meta[item_id]
            title = item_info.get('title', '')
            category = item_info.get('main_cat', 'Unknown')
        else:
            title = f"Item {item_id}"
            category = "Unknown"

        train_items_info.append({
            'item_id': item_id,
            'title': title,
            'category': category
        })
        user_profile_texts.append(f"{title} {category}")
    
    # Create user profile text for retrieval
    user_profile_text = " ".join(user_profile_texts)

    # create sample for fewshot ranking
    prompt_sample = ''
    if sample_user_list is not None:
        prompt_sample = 'Learn from the following examples:\n'
        for i in range(len(sample_user_list)):
            sample_user_data = sample_user_list[i]
            sample_train_items_info = []
            for item_id in sample_user_data['train'][-10:]:
                if item_id in items_meta:
                    item_info = items_meta[item_id]
                    title = item_info.get('title', '')
                    category = item_info.get('main_cat', 'Unknown')
                    
                    sample_train_items_info.append({
                        'item_id': item_id,
                        'title': title,
                        'category': category
                    })
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
                    candidate_items_info_sample.append({
                        'item_id': item_id,
                        'title': item_info.get('title', ''),
                        'category': item_info.get('main_cat', 'Unknown')
                    })
                else:
                    candidate_items_info_sample.append({
                        'item_id': item_id,
                        'title': f'Item {item_id}',
                        'category': 'Unknown'
                    })
            prompt_sample += f"""
            Example {i+1}:
            Other user Recent History: {sample_user_profile_text}
            Candidate Items: {json.dumps(candidate_items_info_sample, indent=2)}
            You should set the true items "{json.dumps(ground_truth_sample_fewshot[i], indent=2)}" at the top of the ranking.\n
            """
        # user_profile_text += " " + sample_user_profile_text
    
    # Combine ground truth and negatives as candidates
    if fixed_candidates is not None:
        candidates = list(fixed_candidates)
    else:
        candidates = ground_truth + negatives
        random.shuffle(candidates)
    candidate_items_info = []
    for item_id in candidates:
        item_state = item_states.get(item_id) if item_states else None
        if eval_variant == "v2" and item_state is not None:
            candidate_items_info.append({
                'item_id': item_id,
                'title': item_state.title,
                'category': item_state.category,
                'memory': item_state.memory
            })
        elif eval_variant == "v2" and item_id in items_meta:
            item_info = items_meta[item_id]
            title = item_info.get('title', '')
            category = item_info.get('main_cat', 'Unknown')
            candidate_items_info.append({
                'item_id': item_id,
                'title': title,
                'category': category,
                'memory': f"The item is called '{title}'. The category is: '{category}'."
            })
        elif eval_variant == "v2":
            candidate_items_info.append({
                'item_id': item_id,
                'title': f'Item {item_id}',
                'category': 'Unknown',
                'memory': "No item memory available."
            })
        elif item_state is not None:
            candidate_items_info.append({
                'item_id': item_id,
                'title': item_state.title,
                'category': item_state.category
            })
        elif item_id in items_meta:
            item_info = items_meta[item_id]
            candidate_items_info.append({
                'item_id': item_id,
                'title': item_info.get('title', ''),
                'category': item_info.get('main_cat', 'Unknown')
            })
        else:
            candidate_items_info.append({
                'item_id': item_id,
                'title': f'Item {item_id}',
                'category': 'Unknown'
            })
    
    # Retrieve relevant memories based on user profile
    if use_memory:
        retrieved_memories = memory_system.retrieve_relevant_memories(user_profile_text, k=k_memories)
    else:
        retrieved_memories = None
    # Use LLM to rank candidates
    predictions = memory_system.llm_ranking(
        train_items_info,
        candidate_items_info,
        retrieved_memories,
        prompt_sample,
        user_memory=user_state.short_term_memory if (eval_variant == "v2" and user_state) else None,
        eval_variant=eval_variant
    )
    baseline_metric = {
        'recall@5': calculate_recall_at_k(candidates, ground_truth, 5),
        'recall@10': calculate_recall_at_k(candidates, ground_truth, 10),
        'recall@20': calculate_recall_at_k(candidates, ground_truth, 20),
        'ndcg@5': calculate_ndcg_at_k(candidates, ground_truth, 5),
        'ndcg@10': calculate_ndcg_at_k(candidates, ground_truth, 10),
        'ndcg@20': calculate_ndcg_at_k(candidates, ground_truth, 20),
    }
    # Calculate metrics
    metrics = {
        'recall@5': calculate_recall_at_k(predictions, ground_truth, 5),
        'recall@10': calculate_recall_at_k(predictions, ground_truth, 10),
        'recall@20': calculate_recall_at_k(predictions, ground_truth, 20),
        'ndcg@5': calculate_ndcg_at_k(predictions, ground_truth, 5),
        'ndcg@10': calculate_ndcg_at_k(predictions, ground_truth, 10),
        'ndcg@20': calculate_ndcg_at_k(predictions, ground_truth, 20),
    }
    
    return baseline_metric,metrics, candidates, predictions, ground_truth

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment configuration")

    # Basic config
    parser.add_argument("--data_name", type=str, default="Video_Game")

    parser.add_argument("--use_memory", action="store_true", default=True)
    # parser.add_argument("--no_use_memory", action="store_false", dest="use_memory")

    parser.add_argument("--LOAD_SAVED_MEMORY", action="store_true", default=False)

    # Hyperparameters for training
    parser.add_argument("--wo_evolving", action="store_true", default=True)
    parser.add_argument("--wo_link", action="store_true", default=False)

    parser.add_argument("--max_evolutions_per_memory", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--link_size", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=1)

    # Hyperparameter for ranking
    parser.add_argument("--k_memories", type=int, default=1)
    parser.add_argument("--eval_variants", type=str, default="both", help="v1, v2, or both")

    # Hyperparameter for few-shot ranking (LLM ranking)
    parser.add_argument("--fewshot_ranking", action="store_true", default=False)
    parser.add_argument("--k_shot", type=int, default=3)

    # Other
    parser.add_argument("--number_of_users", type=int, default=100)
    return parser.parse_args()

def main():
    # Paths to data files
    args = parse_args()

    data_name = args.data_name
    use_memory = args.use_memory
    LOAD_SAVED_MEMORY = args.LOAD_SAVED_MEMORY

    wo_evolving = args.wo_evolving
    wo_link = args.wo_link
    max_evolutions_per_memory = args.max_evolutions_per_memory
    window_size = args.window_size
    link_size = args.link_size
    max_iterations = args.max_iterations

    k_memories = args.k_memories
    fewshot_ranking = args.fewshot_ranking
    k_shot = args.k_shot
    eval_variants_raw = args.eval_variants.lower().strip()
    if eval_variants_raw == "both":
        eval_variants = ["v1", "v2"]
    else:
        eval_variants = [v.strip() for v in eval_variants_raw.split(",") if v.strip()]
    valid_variants = {"v1", "v2"}
    if not eval_variants or any(v not in valid_variants for v in eval_variants):
        raise ValueError(f"Invalid --eval_variants={args.eval_variants}. Use v1, v2, or both.")

    number_of_users = args.number_of_users

    # Resolve paths from the repo root by default so the copied project is self-contained.
    base_dir = os.getenv(
        "AGENTICREC_REPO_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
    data_root = os.getenv("AGENTICREC_DATA_ROOT", os.path.join(base_dir, "data"))
    eval_root = os.getenv("AGENTICREC_EVAL_ROOT", os.path.join(base_dir, "evaluation_results"))
    memory_root = os.getenv("AGENTICREC_MEMORY_ROOT", os.path.join(base_dir, "agent_memory"))

    items_path = os.path.join(data_root, data_name, "items.json")
    sequences_path = os.path.join(data_root, data_name, "user_sequences_10.json")
    negatives_path = os.path.join(data_root, data_name, "user_negatives_10.json")
    
    if use_memory:
        if wo_evolving:
            output_base = f"nuser{number_of_users}_fail_interactions_no_evolving_k{k_memories}_iter{max_iterations}_memory"
            memory_file_path = os.path.join(memory_root, data_name, f"nuser{number_of_users}_fail_interactions_no_evolving_iter{max_iterations}.json")
        elif wo_link:
            output_base = f"nuser{number_of_users}_fail_interactions_no_link_k{k_memories}_iter{max_iterations}_memory_maxevolution{str(max_evolutions_per_memory)}"
            memory_file_path = os.path.join(memory_root, data_name, f"nuser{number_of_users}_fail_interactions_no_link_iter{max_iterations}_maxevolution{str(max_evolutions_per_memory)}.json")
        else:
            output_base = f"nuser{number_of_users}_global_fail_interactions_{k_memories}_iter{max_iterations}_link{link_size}_memory_maxevolution{str(max_evolutions_per_memory)}"
            memory_file_path = os.path.join(memory_root, data_name, f"nuser{number_of_users}_global_fail_interactions_iter{max_iterations}_link{link_size}_maxevolution{str(max_evolutions_per_memory)}.json")
    else:
        if fewshot_ranking:
            output_base = f"nuser{number_of_users}_fewshot_{k_shot}_users_ranking_no_memory"
        else:
            output_base = f"nuser{number_of_users}_zeroshot_users_ranking_no_memory"
    # Load data
    items_meta, user_sequences, user_negatives = load_data(
        items_path, sequences_path, negatives_path
    )

    print(f"Total users loaded: {len(user_sequences)}")
    
    # Get first 100 users
    user_ids = list(user_sequences.keys())[: number_of_users]
    if not use_memory and fewshot_ranking:
        sample_user_ids = list(user_sequences.keys())[number_of_users:]
    
    global_memory = RecommendationMemorySystem(use_gemini_embeddings=True)
    user_states: Dict[str, AgentCFUserState] = {}
    item_states = init_agentcf_item_states(items_meta)
    state_file_path = os.path.join(
        memory_root,
        data_name,
        f"nuser{number_of_users}_agent_states_iter{max_iterations}.json",
    )

    if use_memory:
        if LOAD_SAVED_MEMORY and os.path.exists(memory_file_path):
            print("\n" + "="*80)
            print("LOADING SAVED MEMORY SYSTEM")
            print("="*80)
            global_memory.load_memory(memory_file_path)
            if os.path.exists(state_file_path):
                user_states, loaded_item_states = load_agent_states(state_file_path)
                # Keep metadata-complete item states while overriding known updated memories.
                if loaded_item_states:
                    item_states = loaded_item_states
            else:
                print(f"⚠ No saved agent states found at {state_file_path}. Using fresh user/item memories.")
        else:
            print("\n" + "="*80)
            print("PHASE 1: TRAINING WITH CROSS-USER EVOLVING ONLY")
            print("="*80)
            
            global_memory = RecommendationMemorySystem(use_gemini_embeddings=False)
            
            # shuffled_user_ids = user_ids.copy()
            # random.shuffle(shuffled_user_ids)
            
            for user_id in tqdm(user_ids, desc="Cross-user Training"):
                user_data = user_sequences[user_id]
                print(f"\nProcessing user {user_id} ({len(user_data['train'])} interactions)")
                
                # Tạo temp system chỉ để generate new memories
                # temp_system = RecommendationMemorySystem(use_gemini_embeddings=False)
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

                # reset memory data only
                temp_system.behavior_memories = []
                temp_system.user_interaction_history = []
                temp_system.next_thought_id = 0
                
                # Chỉ tạo new memories từ user này (không evolve nội bộ)
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
                    # Offset thought_id của memory mới
                    new_mem.thought_id += (max_global_id + 1)
                    
                    # Link với global (linked_ids là id cũ trong global)
                    if not wo_evolving:
                        try:
                            linked_ids = global_memory.link_behavior_memories(new_mem, k=link_size, wo_link=wo_link)
                            new_mem.links = linked_ids  # vẫn là id cũ, đúng
                            
                            # Evolve global dựa trên new_mem
                            global_memory.evolve_behavior_memories(new_mem, linked_ids, max_evolutions_per_memory=max_evolutions_per_memory)
                        except Exception as e:
                            print(f"\nError evolving memory for user {user_id}: {e}")
                    
                    # Add vào global
                    global_memory.behavior_memories.append(new_mem)
                    
                    # Update next_id
                    global_memory.next_thought_id = new_mem.thought_id + 1
                
                print(f"  → Global memory pool now has {len(global_memory.behavior_memories)} memories")
                global_memory.save_memory(memory_file_path, format='json')
                save_agent_states(user_states, item_states, state_file_path)
                print(f"  → Overwritten common global memory file: {memory_file_path}")
        
        print("\n" + "="*80)
        print("SAVING GLOBAL CROSS-USER EVOLVING MEMORY")
        print("="*80)
        global_memory.save_memory(memory_file_path, format='json')
        save_agent_states(user_states, item_states, state_file_path)
        global_memory.print_evolution_report()
        stats = global_memory.get_evolution_statistics()
        stats_file = memory_file_path.replace('.json', '_evolution_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Evolution statistics saved to {stats_file}")
    elif LOAD_SAVED_MEMORY:
        if os.path.exists(state_file_path):
            user_states, loaded_item_states = load_agent_states(state_file_path)
            if loaded_item_states:
                item_states = loaded_item_states
        else:
            print(f"⚠ LOAD_SAVED_MEMORY is enabled but no saved agent states at {state_file_path}")
            

    
    # PHASE 2: EVALUATE ON VALIDATION SET
    print("\n" + "="*80)
    print("PHASE 2: VALIDATION SET EVALUATION")
    print("="*80)

    baseline_metrics = {
        'recall@5': [], 'recall@10': [], 'recall@20': [],
        'ndcg@5': [], 'ndcg@10': [], 'ndcg@20': []
    }
    val_metrics_by_variant = {
        variant: {
            'recall@5': [], 'recall@10': [], 'recall@20': [],
            'ndcg@5': [], 'ndcg@10': [], 'ndcg@20': []
        }
        for variant in eval_variants
    }
    all_user_results_by_variant = {variant: [] for variant in eval_variants}
    
    if not use_memory:
        global_memory = RecommendationMemorySystem(use_gemini_embeddings=True)
    for user_id in tqdm(user_ids, desc="Validation"):
        try:
            user_data = user_sequences[user_id]
            negative_data = user_negatives.get(user_id, {})
            fixed_candidates = user_data['test'] + negative_data.get('test_neg', [])
            random.shuffle(fixed_candidates)

            # sample_user_id = random.choice(sample_user_ids) if not use_memory and fewshot_ranking else None
            sample_user_id_list = random.sample(sample_user_ids, k_shot) if not use_memory and fewshot_ranking else None
            if sample_user_id_list is not None:
                sample_user_list = []
                negative_data_sample_list = []
                for id in sample_user_id_list:
                    sample_user_data = user_sequences[id] if id else None
                    negative_data_sample = user_negatives.get(id, {}) if id else None   
                    sample_user_list.append(sample_user_data)
                    negative_data_sample_list.append(negative_data_sample)
            else:
                sample_user_list = None
                negative_data_sample_list = None
            # print(sample_user_data)

            baseline_added = False
            for variant in eval_variants:
                baseline_metric, metrics, candidates, predictions, ground_truth = evaluate_user(
                    user_data, negative_data,
                    items_meta,
                    global_memory,
                    eval_type='test',
                    use_memory=use_memory,
                    k_memories=k_memories,
                    sample_user_list=sample_user_list,
                    negative_data_sample_list=negative_data_sample_list,
                    user_state=get_or_create_user_state(user_states, user_id),
                    item_states=item_states,
                    eval_variant=variant,
                    fixed_candidates=fixed_candidates,
                )
                all_user_results_by_variant[variant].append({
                    "user_id": user_id,
                    "ground_truth": ground_truth,
                    "candidates": candidates,
                    "predictions": predictions,
                    "metrics": metrics,
                    "baseline_metrics": baseline_metric
                })
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
            output_file=output_file
        )

    # Print validation results
    print("\nValidation Results:")
    print("-" * 80)
    for metric in ['recall@5', 'recall@10', 'recall@20','ndcg@5', 'ndcg@10', 'ndcg@20']:
        if len(baseline_metrics[metric]) > 0:
            mean_val = np.mean(baseline_metrics[metric])
            print(f"Baseline {metric:10s}: {mean_val:.4f}")
        else:
            print(f"Baseline {metric:10s}: N/A")
    for variant in eval_variants:
        print("-" * 80)
        print(f"Variant {variant.upper()}:")
        for metric in ['recall@5', 'recall@10', 'recall@20','ndcg@5', 'ndcg@10', 'ndcg@20']:
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
            for metric in ['recall@5', 'recall@10', 'recall@20','ndcg@5', 'ndcg@10', 'ndcg@20']
        },
        "variant_metrics": {
            variant: {
                metric: (
                    float(np.mean(val_metrics_by_variant[variant][metric]))
                    if len(val_metrics_by_variant[variant][metric]) > 0 else None
                )
                for metric in ['recall@5', 'recall@10', 'recall@20','ndcg@5', 'ndcg@10', 'ndcg@20']
            }
            for variant in eval_variants
        }
    }
    summary_path = os.path.join(eval_root, data_name, f"{output_base}.summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved AgentMemCF summary to {summary_path}")

if __name__ == "__main__":
    main()
