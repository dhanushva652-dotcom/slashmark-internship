from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    fraud_cost: float = 25.0
    false_alarm_cost: float = 1.0
