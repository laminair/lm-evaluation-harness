"""
Stateful router example with classifier-based routing and optional energy monitoring.

This demonstrates the recommended pattern for implementing stateful routers that:
- Load and own their models from YAML config
- Implement classifier-based routing on GPU
- Support gradient accumulation for efficient training
- Optionally integrate Zeus for energy monitoring
- Support checkpointing for reproducibility

Usage:
    from examples.stateful_router import StatefulRouter
    from lm_eval.models.router import RouterLM

    # Create router with optional energy monitoring
    router = StatefulRouter.from_yaml(
        config_path="config.yaml",
        device="cuda",
        monitor_energy=True,  # Requires: pip install zeus-ml
    )

    # Create RouterLM wrapper
    router_lm = RouterLM.from_router(router)

    # Evaluate
    results = evaluator.simple_evaluate(model=router_lm, tasks=["hellaswag"])

    # Get statistics (includes energy/latency if monitored)
    stats = router.get_stats()

    # Checkpoint
    checkpoint = router.get_state()

Config YAML format:
    models:
      small:
        type: vllm
        pretrained: meta-llama/Llama-3.2-1B
        tensor_parallel_size: 1
        metadata:
          params_billion: 1
          cost_per_1k_tokens: 0.0001
      large:
        type: vllm
        pretrained: meta-llama/Llama-3.1-8B
        tensor_parallel_size: 2
        metadata:
          params_billion: 8
          cost_per_1k_tokens: 0.0005

    router:
      weights: {small: 0.5, large: 0.5}
      learning_rate: 0.01
      exploration_rate: 0.15
      classifier:
        input_dim: 768
        hidden_dim: 256
        accum_steps: 32
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
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


logger = logging.getLogger(__name__)


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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class StatefulRouter:
    """
    Stateful router with classifier-based routing and optional energy monitoring.

    Features:
    - Model loading from YAML config
    - Simple classifier on GPU for routing decisions
    - Epsilon-greedy exploration
    - Gradient accumulation for efficient training
    - Optional Zeus energy monitoring
    - Latency tracking (always on)
    - State checkpointing

    Attributes:
        models: Dict of loaded LM instances
        model_info: Dict of model metadata from config
        weights: Routing weights (can be updated by learning)
        classifier: Neural network for routing decisions
        outcomes: List of recorded outcomes for analysis
    """

    def __init__(
        self,
        config_path: str,
        device: str = "cuda",
        monitor_energy: bool = False,
        gpu_indices: list[int] | None = None,
    ):
        """
        Initialize the router.

        Args:
            config_path: Path to YAML config file
            device: Device for classifier ("cuda" or "cpu")
            monitor_energy: Enable Zeus energy monitoring (requires zeus-ml)
            gpu_indices: GPU indices for energy monitoring (default: all)
        """
        self.config = RouterConfig.from_yaml(config_path)
        self.device = device

        self.models: dict[str, LM] = self._load_models()

        self.model_info: dict[str, dict[str, Any]] = {
            model_id: model_cfg.get("metadata", {})
            for model_id, model_cfg in self.config.models.items()
        }

        router_cfg = self.config.router
        self.weights: dict[str, float] = dict(router_cfg.get("weights", {}))
        self.learning_rate: float = router_cfg.get("learning_rate", 0.01)
        self.exploration_rate: float = router_cfg.get("exploration_rate", 0.15)

        self._normalize_weights()

        classifier_cfg = router_cfg.get("classifier", {})
        self._accum_steps: int = classifier_cfg.get("accum_steps", 32)
        self._step_count: int = 0

        num_models = len(self.models)
        input_dim = classifier_cfg.get("input_dim", 768)
        hidden_dim = classifier_cfg.get("hidden_dim", 256)

        self.classifier = SimpleClassifier(input_dim, hidden_dim, num_models).to(device)
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=self.learning_rate
        )

        self.model_id_to_idx = {
            model_id: i for i, model_id in enumerate(self.models.keys())
        }
        self.idx_to_model_id = {
            i: model_id for model_id, i in self.model_id_to_idx.items()
        }

        self.outcomes: list[dict[str, Any]] = []

        self._energy_monitor = None
        self._monitor_energy = monitor_energy
        self._gpu_indices = gpu_indices

        if monitor_energy:
            try:
                from zeus.monitor import ZeusMonitor

                self._energy_monitor = ZeusMonitor(gpu_indices=gpu_indices)
                logger.info("Zeus energy monitoring enabled")
            except ImportError:
                logger.warning(
                    "zeus-ml not installed. Energy monitoring disabled. "
                    "Install with: pip install zeus-ml"
                )
                self._monitor_energy = False

        logger.info(
            f"StatefulRouter initialized with {len(self.models)} models: "
            f"{list(self.models.keys())}"
        )

    @classmethod
    def from_yaml(
        cls,
        config_path: str,
        device: str = "cuda",
        monitor_energy: bool = False,
        gpu_indices: list[int] | None = None,
    ) -> "StatefulRouter":
        """Create router from YAML config file."""
        return cls(config_path, device, monitor_energy, gpu_indices)

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
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

            model_args = {
                k: v for k, v in model_cfg.items() if k not in ("type", "metadata")
            }

            model_cls = get_model(model_type)

            try:
                model = model_cls.create_from_arg_obj(model_args, {})
            except TypeError:
                model = model_cls(**model_args)

            models[model_id] = model
            logger.info(f"Loaded model '{model_id}': {model_type}")

        return models

    def _extract_features(
        self, request: "Instance", context: RoutingContext
    ) -> torch.Tensor:
        """
        Extract features from request for classifier input.

        Override this method to implement custom feature extraction,
        e.g., using sentence embeddings, prompt length, task type, etc.

        Default implementation uses simple heuristic features.
        """
        prompt = str(request.args[0]) if request.args else ""

        task_hash = hash(context.task_name or "") % 10000 / 10000.0
        type_hash = hash(request.request_type) % 10000 / 10000.0
        length_feature = min(len(prompt) / 2000.0, 1.0)

        features = [length_feature, task_hash, type_hash]

        while len(features) < 768:
            features.append(0.0)

        return torch.tensor(features[:768], dtype=torch.float32, device=self.device)

    def seed(self, seed: int) -> None:
        """
        Seed all random number generators for reproducibility.

        Args:
            seed: Random seed value
        """
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        logger.info(f"Router seeded with {seed}")

    def route(
        self,
        request: "Instance",
        context: RoutingContext,
        state: dict[str, Any],
    ) -> str | RoutingDecision:
        """
        Make a routing decision using classifier + epsilon-greedy exploration.

        Args:
            request: The evaluation request to route
            context: Routing context with metadata
            state: State dict (unused for instance methods)

        Returns:
            RoutingDecision with primary model and shadow models for learning
        """
        model_ids = list(self.models.keys())

        if random.random() < self.exploration_rate:
            primary = random.choice(model_ids)
        else:
            with torch.no_grad():
                features = self._extract_features(request, context)
                logits = self.classifier(features.unsqueeze(0))
                idx = torch.argmax(logits, dim=1).item()
                primary = self.idx_to_model_id[idx]

        shadows = [m for m in model_ids if m != primary]

        return RoutingDecision(
            primary_model=primary,
            shadow_models=shadows,
            metadata={
                "method": "classifier",
                "exploration": random.random() < self.exploration_rate,
            },
        )

    def update(self, event: OutcomeEvent, state: dict[str, Any]) -> None:
        """
        Process outcome, train classifier with gradient accumulation.

        Args:
            event: Outcome event with results from all models
            state: State dict (unused for instance methods)
        """
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

        if event.all_correct:
            best_model = max(event.all_correct, key=event.all_correct.get)
            target_idx = self.model_id_to_idx[best_model]

            features = self._extract_features(
                event.request,
                RoutingContext(
                    request_type=event.request.request_type,
                    task_name=event.task_name,
                    doc_id=event.doc_id,
                    doc=event.doc,
                    arguments=event.request.args,
                ),
            )

            logits = self.classifier(features.unsqueeze(0))
            target = torch.tensor([target_idx], dtype=torch.long, device=self.device)
            loss = F.cross_entropy(logits, target)

            (loss / self._accum_steps).backward()
            self._step_count += 1

            if self._step_count % self._accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.outcomes.append(outcome_record)

    def get_state(self) -> dict[str, Any]:
        """
        Return state for checkpointing.

        Returns:
            Dict containing weights, classifier state, optimizer state, and outcomes
        """
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
        """
        Restore state from checkpoint.

        Args:
            state: Dict previously returned by get_state()
        """
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
        logger.info("Router state restored from checkpoint")

    def get_stats(self) -> dict[str, Any]:
        """
        Get router statistics.

        Returns:
            Dict with accuracy, energy, latency, and weight statistics
        """
        total = len(self.outcomes)
        correct = sum(1 for o in self.outcomes if o.get("primary_correct", False))

        total_energy = sum(
            sum(o.get("energy_joules", {}).values()) for o in self.outcomes
        )
        total_latency = sum(
            sum(o.get("latency_ms", {}).values()) for o in self.outcomes
        )

        per_model_energy: dict[str, float] = {}
        per_model_latency: dict[str, float] = {}
        per_model_count: dict[str, int] = {}

        for o in self.outcomes:
            for model_id, energy in o.get("energy_joules", {}).items():
                per_model_energy[model_id] = per_model_energy.get(model_id, 0) + energy
            for model_id, latency in o.get("latency_ms", {}).items():
                per_model_latency[model_id] = (
                    per_model_latency.get(model_id, 0) + latency
                )
            model = o.get("primary_model")
            if model:
                per_model_count[model] = per_model_count.get(model, 0) + 1

        return {
            "weights": dict(self.weights),
            "total_decisions": total,
            "correct_decisions": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "total_energy_joules": total_energy,
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / total if total > 0 else 0.0,
            "per_model_energy_joules": per_model_energy,
            "per_model_latency_ms": per_model_latency,
            "per_model_count": per_model_count,
        }

    @property
    def model_ids(self) -> list[str]:
        """Get list of model IDs."""
        return list(self.models.keys())

    def get_model_energy(
        self, model_id: str, request: "Instance", method: str
    ) -> tuple[Any, float]:
        """
        Execute a request on a single model with energy measurement.

        Use this when you need to measure energy for each model separately.

        Args:
            model_id: ID of the model to execute on
            request: The request to execute
            method: Method name ("loglikelihood", "generate_until", etc.)

        Returns:
            Tuple of (response, energy_joules)
        """
        import time

        model = self.models[model_id]

        energy = 0.0

        if self._energy_monitor:
            self._energy_monitor.begin_window(f"single_{model_id}")

        start_time = time.perf_counter()
        response = getattr(model, method)([request])[0]
        end_time = time.perf_counter()

        if self._energy_monitor:
            measurement = self._energy_monitor.end_window(f"single_{model_id}")
            energy = sum(measurement.energy.values())

        latency = (end_time - start_time) * 1000

        return response, energy, latency
