import json
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .io_utils import get_or_create_user_state
from .memory_system import RecommendationMemorySystem
from .models import AgentCFItemState, AgentCFUserState, BehaviorMemory, UserInteraction


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
    if raw_output.startswith("```"):
        raw_output = re.sub(r"^```[a-zA-Z]*", "", raw_output)
        raw_output = raw_output.rstrip("```").strip()

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    json_str = match.group(0)

    try:
        data = json.loads(json_str)
        item1_desc = data["item_1"].strip()
        item2_desc = data["item_2"].strip()
    except Exception as e:
        raise ValueError(f"Invalid JSON output from LLM: {e}")

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
    train_items = user_data["train"][-30:]
    if len(train_items) == 0:
        return []

    user_state = get_or_create_user_state(user_states, user_id)
    all_item_ids = list(item_states.keys())
    if not all_item_ids:
        return []
    blocked_items: Set[str] = set(train_items) | set(user_state.interaction_history)

    def sample_negative_item_fast(pos_item_id: str, max_tries: int = 64) -> Optional[str]:
        total_items = len(all_item_ids)
        if total_items <= 1:
            return None
        for _ in range(max_tries):
            cand = random.choice(all_item_ids)
            if cand != pos_item_id and cand not in blocked_items:
                return cand
        start = random.randrange(total_items)
        for offset in range(total_items):
            cand = all_item_ids[(start + offset) % total_items]
            if cand != pos_item_id and cand not in blocked_items:
                return cand
        return None

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
