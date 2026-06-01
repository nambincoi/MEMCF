from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class UserInteraction:
    """Represents a single user-item interaction."""

    item_id: str
    item_name: str
    item_category: str
    action_type: str
    rating: Optional[float] = None
    timestamp: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BehaviorMemory:
    """Represents a generalized thought about user behavior patterns."""

    thought_id: int
    interaction_sequence: List[UserInteraction]
    behavior_explanation: str
    pattern_description: str
    keywords: List[str]
    embedding: np.ndarray
    links: List[int] = field(default_factory=list)
    timestamp: str = None
    evolution_count: int = 0
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    max_evolutions: Optional[int] = None
    last_evolved_timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def can_evolve(self) -> bool:
        if self.max_evolutions is None:
            return True
        return self.evolution_count < self.max_evolutions

    def record_evolution(
        self,
        update_type: str,
        old_values: Dict[str, Any],
        new_values: Dict[str, Any],
        reasoning: str,
    ) -> None:
        self.evolution_count += 1
        self.last_evolved_timestamp = datetime.now().isoformat()
        self.evolution_history.append(
            {
                "evolution_number": self.evolution_count,
                "timestamp": self.last_evolved_timestamp,
                "update_type": update_type,
                "old_values": old_values,
                "new_values": new_values,
                "reasoning": reasoning,
            }
        )

    def to_dict(self):
        data = asdict(self)
        data["embedding"] = self.embedding.tolist()
        data["interaction_sequence"] = [asdict(i) for i in self.interaction_sequence]
        return data

    @classmethod
    def from_dict(cls, data):
        data["embedding"] = np.array(data["embedding"])
        data["interaction_sequence"] = [UserInteraction(**i) for i in data["interaction_sequence"]]
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
