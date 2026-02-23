# RouterLM Programmatic API Implementation Plan

## Overview

Refactor RouterLM to support stateful router classes that own their models, with optional Zeus energy monitoring integration.

## Files to Modify

### 1. `lm_eval/api/router.py` - Add energy/latency to OutcomeEvent

```python
@dataclass
class OutcomeEvent:
    # ... existing fields ...
    
    latency_ms: dict[str, float] = field(default_factory=dict)      # model_id -> ms
    energy_joules: dict[str, float] = field(default_factory=dict)   # model_id -> joules
```

### 2. `lm_eval/models/router.py` - Major Refactor

#### Remove
- `from_callbacks()` factory method

#### Modify
- `_execute_on_models()` - Add latency measurement, optional Zeus energy monitoring
- `RoutingDecisionInternal` - Add `latencies` and `energies` fields

#### Add
- `from_router()` factory method
- `_energy_monitor` attribute (optional ZeusMonitor)
- Energy monitoring initialization/destruction

#### Key Changes

```python
class RouterLM(LM):
    def __init__(self, config_path: str, ...):
        # ... existing __init__ ...
        self._energy_monitor = None  # Optional Zeus integration
        self._router_instance = None  # Reference to router for state management
    
    @classmethod
    def from_router(
        cls,
        router: Any,
        monitor_energy: bool = False,
        gpu_indices: list[int] | None = None,
    ) -> "RouterLM":
        """
        Create RouterLM from a router instance that owns its models.
        
        Args:
            router: Router instance with:
                - .models: dict[str, LM] - loaded model instances
                - .model_info: dict[str, dict] - model metadata (optional)
                - .route(request, context, state) -> str | RoutingDecision
                - .update(event, state) -> None
                - .get_state() -> dict (optional, for checkpointing)
                - .set_state(state) (optional, for checkpointing)
            monitor_energy: Enable Zeus energy monitoring
            gpu_indices: GPU indices for energy monitoring (default: all)
        
        Returns:
            Configured RouterLM instance
        """
        instance = cls.__new__(cls)
        super(RouterLM, instance).__init__()
        
        # Extract from router
        instance._models = dict(router.models)
        instance._routing_callback = router.route
        instance._outcome_callback = router.update
        instance._router_instance = router
        
        # Set defaults
        instance._primary_model_default = list(router.models.keys())[0]
        instance._shadow_mode = "none"
        instance._state = {}
        instance._adaptive = False
        instance._pending_decisions = {}
        instance.config_path = ""
        instance.config = None
        
        # Energy monitoring
        instance._energy_monitor = None
        if monitor_energy:
            try:
                from zeus.monitor import ZeusMonitor
                instance._energy_monitor = ZeusMonitor(gpu_indices=gpu_indices)
            except ImportError:
                import warnings
                warnings.warn("zeus not installed. Energy monitoring disabled.")
        
        return instance
    
    def _execute_on_models(
        self,
        request: Instance,
        decision: RoutingDecisionInternal,
        method: str,
    ) -> Any:
        """Execute a request on primary and shadow models with timing/energy."""
        import time
        
        all_responses = {}
        latencies = {}
        energies = {}
        
        model_ids = [decision.primary_model] + decision.shadow_models
        
        for model_id in model_ids:
            model = self._models[model_id]
            
            # Start measurement
            start_time = time.perf_counter()
            if self._energy_monitor:
                self._energy_monitor.begin_window(f"inference_{model_id}")
            
            # Execute
            response = getattr(model, method)([request])[0]
            
            # End measurement
            end_time = time.perf_counter()
            if self._energy_monitor:
                measurement = self._energy_monitor.end_window(f"inference_{model_id}")
                # Sum energy across all GPUs
                total_energy = sum(measurement.energy.values())
                energies[model_id] = total_energy
            
            all_responses[model_id] = response
            latencies[model_id] = (end_time - start_time) * 1000  # ms
        
        decision.all_responses = all_responses
        decision.latencies = latencies
        decision.energies = energies
        
        return all_responses[decision.primary_model]
```

### 3. `examples/stateful_router.py` - NEW FILE

Complete skeleton with:
- Model loading from YAML
- Simple classifier on GPU
- Gradient accumulation
- Energy/latency tracking
- State checkpointing
- Zeus integration example

```python
"""
Stateful router example with classifier-based routing.

Usage:
    router = StatefulRouter.from_yaml("config.yaml")
    router_lm = RouterLM.from_router(router, monitor_energy=True)
    results = evaluator.simple_evaluate(model=router_lm, tasks=["hellaswag"])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from lm_eval.api.model import LM
from lm_eval.api.registry import get_model
from lm_eval.api.router import OutcomeEvent, RoutingContext, RoutingDecision

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


@dataclass
class RouterConfig:
    models: dict[str, dict[str, Any]]
    router: dict[str, Any]
    
    @classmethod
    def from_yaml(cls, path: str) -> "RouterConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            models=data.get("models", {}),
            router=data.get("router", {}),
        )


class SimpleClassifier(nn.Module):
    """Simple MLP classifier for routing decisions."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_models: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class StatefulRouter:
    """
    Example stateful router with:
    - Model loading from YAML config
    - Simple classifier on GPU for routing
    - Gradient accumulation for efficient training
    - Energy/latency tracking via OutcomeEvent
    - State checkpointing
    
    Config YAML format:
        models:
          small:
            type: vllm
            pretrained: meta-llama/Llama-3.2-1B
            metadata:
              params_billion: 1
              cost_per_1k_tokens: 0.0001
          large:
            type: vllm
            pretrained: meta-llama/Llama-3.1-8B
            metadata:
              params_billion: 8
              cost_per_1k_tokens: 0.0005
        
        router:
          weights: {small: 0.5, large: 0.5}
          learning_rate: 0.1
          exploration_rate: 0.15
          classifier:
            input_dim: 768
            hidden_dim: 256
            accum_steps: 32
    """
    
    def __init__(
        self,
        config_path: str,
        device: str = "cuda",
        monitor_energy: bool = False,
    ):
        self.config = RouterConfig.from_yaml(config_path)
        self.device = device
        
        # Load models
        self.models: dict[str, LM] = self._load_models()
        
        # Extract model info
        self.model_info: dict[str, dict[str, Any]] = {
            model_id: model_cfg.get("metadata", {})
            for model_id, model_cfg in self.config.models.items()
        }
        
        # Router config
        router_cfg = self.config.router
        self.weights: dict[str, float] = router_cfg.get("weights", {})
        self.learning_rate: float = router_cfg.get("learning_rate", 0.1)
        self.exploration_rate: float = router_cfg.get("exploration_rate", 0.15)
        
        # Normalize weights
        if self.weights:
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Classifier
        classifier_cfg = router_cfg.get("classifier", {})
        self._accum_steps: int = classifier_cfg.get("accum_steps", 32)
        self._step_count: int = 0
        
        num_models = len(self.models)
        input_dim = classifier_cfg.get("input_dim", 768)
        hidden_dim = classifier_cfg.get("hidden_dim", 256)
        
        self.classifier = SimpleClassifier(input_dim, hidden_dim, num_models).to(device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        
        # Model ID to index mapping
        self.model_id_to_idx = {model_id: i for i, model_id in enumerate(self.models.keys())}
        self.idx_to_model_id = {i: model_id for model_id, i in self.model_id_to_idx.items()}
        
        # Outcome tracking
        self.outcomes: list[dict[str, Any]] = []
        
        # Energy monitoring (optional)
        self._energy_monitor = None
        if monitor_energy:
            try:
                from zeus.monitor import ZeusMonitor
                self._energy_monitor = ZeusMonitor()
            except ImportError:
                pass
        
        self._normalize_weights()
    
    @classmethod
    def from_yaml(
        cls,
        config_path: str,
        device: str = "cuda",
        monitor_energy: bool = False,
    ) -> "StatefulRouter":
        return cls(config_path, device, monitor_energy)
    
    def _normalize_weights(self) -> None:
        if self.weights:
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}
    
    def _load_models(self) -> dict[str, LM]:
        """Load all models defined in the config."""
        models = {}
        for model_id, model_cfg in self.config.models.items():
            model_type = model_cfg.get("type")
            if not model_type:
                raise ValueError(f"Model '{model_id}' missing 'type' field")
            
            model_args = {k: v for k, v in model_cfg.items() if k not in ("type", "metadata")}
            
            model_cls = get_model(model_type)
            
            try:
                model = model_cls.create_from_arg_obj(model_args, {})
            except TypeError:
                model = model_cls(**model_args)
            
            models[model_id] = model
            print(f"Loaded model '{model_id}': {model_type}")
        
        return models
    
    def _extract_features(self, request: "Instance", context: RoutingContext) -> torch.Tensor:
        """
        Extract features from request for classifier input.
        
        Override this method to implement custom feature extraction,
        e.g., using sentence embeddings, prompt length, task type, etc.
        """
        # Simple example: use prompt length and hash-based features
        prompt = str(request.args[0]) if request.args else ""
        
        features = [
            len(prompt) / 1000.0,  # Normalized length
            hash(context.task_name or "") % 1000 / 1000.0,  # Task hash
            hash(request.request_type) % 1000 / 1000.0,  # Request type hash
        ]
        
        # Pad to input_dim (768 by default)
        while len(features) < 768:
            features.append(0.0)
        
        return torch.tensor(features[:768], dtype=torch.float32, device=self.device)
    
    def seed(self, seed: int) -> None:
        """Seed all random number generators."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def route(
        self,
        request: "Instance",
        context: RoutingContext,
        state: dict[str, Any],
    ) -> str | RoutingDecision:
        """
        Make a routing decision using classifier + epsilon-greedy exploration.
        """
        import random
        
        model_ids = list(self.models.keys())
        
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            primary = random.choice(model_ids)
        else:
            # Use classifier
            with torch.no_grad():
                features = self._extract_features(request, context)
                logits = self.classifier(features.unsqueeze(0))
                idx = torch.argmax(logits, dim=1).item()
                primary = self.idx_to_model_id[idx]
        
        # Shadow models for learning
        shadows = [m for m in model_ids if m != primary]
        
        return RoutingDecision(
            primary_model=primary,
            shadow_models=shadows,
            metadata={"method": "classifier"},
        )
    
    def update(self, event: OutcomeEvent, state: dict[str, Any]) -> None:
        """
        Process outcome, train classifier with gradient accumulation.
        """
        # Record outcome
        outcome_record = {
            "task": event.task_name,
            "doc_id": event.doc_id,
            "primary_model": event.primary_model,
            "primary_correct": event.primary_correct,
            "all_correct": dict(event.all_correct),
            "latency_ms": dict(event.latency_ms),
            "energy_joules": dict(event.energy_joules),
            "weights_before": dict(self.weights),
        }
        
        # Update weights based on outcome
        if event.primary_correct:
            self.weights[event.primary_model] = self.weights.get(
                event.primary_model, 0.5
            ) * (1 + self.learning_rate)
        
        for shadow_id in event.shadow_models:
            if event.all_correct.get(shadow_id, False):
                self.weights[shadow_id] = self.weights.get(shadow_id, 0.5) * (
                    1 + self.learning_rate * 0.5
                )
        
        self._normalize_weights()
        outcome_record["weights_after"] = dict(self.weights)
        
        # Train classifier with gradient accumulation
        best_model = max(event.all_correct, key=event.all_correct.get)
        target_idx = self.model_id_to_idx[best_model]
        
        # Reconstruct features (in practice, cache these in route())
        features = self._extract_features(event.request, RoutingContext(
            request_type=event.request.request_type,
            task_name=event.task_name,
            doc_id=event.doc_id,
            doc=event.doc,
            arguments=event.request.args,
        ))
        
        logits = self.classifier(features.unsqueeze(0))
        target = torch.tensor([target_idx], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, target)
        
        # Normalize and accumulate
        (loss / self._accum_steps).backward()
        self._step_count += 1
        
        # Update weights every K steps
        if self._step_count % self._accum_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.outcomes.append(outcome_record)
    
    def get_state(self) -> dict[str, Any]:
        """Return state for checkpointing."""
        return {
            "weights": dict(self.weights),
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "classifier_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "outcomes": list(self.outcomes),
            "step_count": self._step_count,
        }
    
    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.weights = dict(state.get("weights", {}))
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.exploration_rate = state.get("exploration_rate", self.exploration_rate)
        self.outcomes = list(state.get("outcomes", []))
        self._step_count = state.get("step_count", 0)
        
        if "classifier_state_dict" in state:
            self.classifier.load_state_dict(state["classifier_state_dict"])
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        
        self._normalize_weights()
    
    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        total = len(self.outcomes)
        correct = sum(1 for o in self.outcomes if o.get("primary_correct", False))
        
        total_energy = sum(
            sum(o.get("energy_joules", {}).values())
            for o in self.outcomes
        )
        total_latency = sum(
            sum(o.get("latency_ms", {}).values())
            for o in self.outcomes
        )
        
        return {
            "weights": dict(self.weights),
            "total_decisions": total,
            "correct_decisions": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "total_energy_joules": total_energy,
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / total if total > 0 else 0.0,
        }
    
    @property
    def model_ids(self) -> list[str]:
        return list(self.models.keys())
```

### 4. `tests/models/test_router.py` - Update Tests

Remove tests for `from_callbacks()`, add tests for:
- `from_router()` factory method
- Energy/latency fields in OutcomeEvent
- State management with router instance

### 5. `docs/router_guide.md` - Update Documentation

Add section on:
- `from_router()` usage
- Energy monitoring with Zeus
- Stateful router implementation guide

## Usage Example

```python
from lm_eval import evaluator
from lm_eval.models.router import RouterLM
from examples.stateful_router import StatefulRouter

# Create router (loads models from YAML)
router = StatefulRouter.from_yaml(
    config_path="config.yaml",
    device="cuda",
    monitor_energy=True,  # Enable Zeus energy monitoring
)

# Create RouterLM wrapper
router_lm = RouterLM.from_router(
    router,
    monitor_energy=True,
    gpu_indices=[0],
)

# Evaluate
results = evaluator.simple_evaluate(
    model=router_lm,
    tasks=["hellaswag"],
    limit=100,
)

# Get statistics
stats = router.get_stats()
print(f"Accuracy: {stats['accuracy']:.2%}")
print(f"Total energy: {stats['total_energy_joules']:.2f} J")
print(f"Avg latency: {stats['avg_latency_ms']:.2f} ms")

# Checkpoint
checkpoint = router.get_state()
import json
with open("checkpoint.json", "w") as f:
    json.dump(checkpoint, f)

# Later, restore
router2 = StatefulRouter.from_yaml("config.yaml")
router2.set_state(checkpoint)
```

## Dependencies

Zeus is optional. Add to `pyproject.toml` as optional dependency:

```toml
[project.optional-dependencies]
energy = ["zeus-ml"]
```

## Implementation Order

1. Update `lm_eval/api/router.py` - Add energy/latency fields
2. Refactor `lm_eval/models/router.py` - Remove `from_callbacks()`, add `from_router()`, add timing
3. Create `examples/stateful_router.py`
4. Update `tests/models/test_router.py`
5. Update `docs/router_guide.md`
