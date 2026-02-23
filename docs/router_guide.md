# RouterLM: Multi-Model Routing for LLM Evaluation

RouterLM enables dynamic routing of evaluation requests between multiple language models. This is useful for:

- **Cost optimization**: Route simple queries to smaller, cheaper models
- **Quality routing**: Use larger models for complex queries
- **A/B testing**: Compare models on the same inputs
- **Online learning**: Adapt routing decisions based on real-time feedback

## Quick Start

Create a router configuration file:

```yaml
# router_config.yaml
models:
  small:
    type: hf
    pretrained: gpt2
    metadata:
      params_billion: 0.124
      cost_per_1k_tokens: 0.00001
  large:
    type: hf
    pretrained: gpt2-medium
    metadata:
      params_billion: 0.355
      cost_per_1k_tokens: 0.00003

routing:
  adaptive: false
```

Run evaluation with the router:

```bash
lm-eval run --router_config router_config.yaml --tasks hellaswag --limit 10
```

## Configuration Reference

### Model Definitions

Define models under the `models` key. Each model requires a `type` and model-specific arguments:

```yaml
models:
  gpt_small:
    type: hf
    pretrained: gpt2
    dtype: float32
    metadata:
      params_billion: 0.124
      cost_per_1k_tokens: 0.00001
  
  gpt_large:
    type: hf
    pretrained: gpt2-medium
    dtype: float32
    metadata:
      params_billion: 0.355
      cost_per_1k_tokens: 0.00003
  
  vllm_model:
    type: vllm
    pretrained: meta-llama/Llama-2-7b-hf
    tensor_parallel_size: 2
    metadata:
      params_billion: 7
      cost_per_1k_tokens: 0.0005
```

The `metadata` section is optional and can contain any information your router needs (e.g., `params_billion`, `cost_per_1k_tokens`).

### Routing Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `adaptive` | bool | `false` | Enable online learning with outcome feedback |
| `routing_callback` | string | none | Python path to routing function |
| `outcome_callback` | string | none | Python path to outcome feedback function |
| `initial_state` | dict | `{}` | Initial state passed to callbacks |

## Writing Routing Callbacks

A routing callback decides which model should handle each request.

### Function Signature

```python
def route_request(
    request: Instance,
    context: RoutingContext,
    state: dict[str, Any]
) -> str | RoutingDecision | dict:
    ...
```

### Parameters

- **`request`**: The `Instance` being evaluated (contains `request_type`, `arguments`, etc.)
- **`context`**: `RoutingContext` with metadata:
  - `request_type`: `"loglikelihood"`, `"generate_until"`, etc.
  - `task_name`: Name of the evaluation task
  - `doc_id`: Document index in the dataset
  - `doc`: The document being evaluated
  - `arguments`: Tuple of request arguments
  - `idx`: Instance index
- **`state`**: Mutable dictionary that persists across all routing calls

### Return Values

The callback can return:

1. **String**: Model ID to route to
   ```python
   return "large_model"
   ```

2. **RoutingDecision**: Model with optional metadata
   ```python
   from lm_eval.api.router import RoutingDecision
   return RoutingDecision(
       model="large_model",
       metadata={"reason": "complex_query"}
   )
   ```

3. **Dict**: Shorthand for RoutingDecision
   ```python
   return {
       "model": "large_model",
       "metadata": {"reason": "complex_query"}
   }
   ```

### Example: Length-Based Router

```python
# my_router/length_router.py

def route_by_length(request, context, state):
    """Route to large model if prompt exceeds threshold."""
    prompt = context.arguments[0] if context.arguments else ""
    threshold = state.get("length_threshold", 500)
    
    if len(prompt) > threshold:
        return "large_model"
    return "small_model"
```

Configuration:
```yaml
routing:
  routing_callback: my_router.length_router:route_by_length
  initial_state:
    length_threshold: 500
```

### Example: Task-Aware Router

```python
# my_router/task_router.py

def route_by_task(request, context, state):
    """Route based on task type."""
    task_routes = state.get("task_routes", {})
    
    if context.task_name in task_routes:
        return task_routes[context.task_name]
    
    # Default routing based on request type
    if context.request_type == "generate_until":
        return "large_model"
    return "small_model"
```

Configuration:
```yaml
routing:
  routing_callback: my_router.task_router:route_by_task
  initial_state:
    task_routes:
      hellaswag: large_model
      arc_easy: small_model
      wikitext: large_model
```

### Example: Epsilon-Greedy Bandit

```python
# my_router/bandit.py
import random

def epsilon_greedy_route(request, context, state):
    """Epsilon-greedy routing based on model scores."""
    scores = state.get("model_scores", {"small": 0.5, "large": 0.5})
    epsilon = state.get("epsilon", 0.1)
    
    if random.random() < epsilon:
        # Explore: random choice
        return random.choice(list(scores.keys()))
    
    # Exploit: best scoring model
    return max(scores, key=scores.get)
```

## Writing Outcome Callbacks

Outcome callbacks receive feedback after each document is evaluated, enabling online learning.

### Function Signature

```python
def on_outcome(event: OutcomeEvent, state: dict[str, Any]) -> None:
    ...
```

### OutcomeEvent Fields

| Field | Type | Description |
|-------|------|-------------|
| `request` | Instance | The original request |
| `task_name` | str | Task name |
| `doc_id` | int | Document index |
| `doc` | dict | Document content |
| `model` | str | Model that was used |
| `response` | Any | Model's response |
| `metrics` | dict | Evaluation metrics |
| `correct` | bool | Whether the answer was correct |
| `routing_metadata` | dict | Metadata from routing decision |
| `latency_ms` | float | Latency in milliseconds |
| `energy_joules` | float | Energy consumption (if monitored) |

### Example: Update Bandit Scores

```python
# my_router/bandit.py

def update_bandit_scores(event, state):
    """Update model scores using exponential moving average."""
    scores = state.setdefault("model_scores", {})
    alpha = state.get("learning_rate", 0.1)
    
    current = scores.get(event.model, 0.5)
    reward = 1.0 if event.correct else 0.0
    scores[event.model] = (1 - alpha) * current + alpha * reward
    
    # Decay epsilon over time
    epsilon = state.get("epsilon", 0.1)
    state["epsilon"] = max(0.01, epsilon * 0.999)
```

Configuration for adaptive routing:
```yaml
routing:
  adaptive: true
  routing_callback: my_router.bandit:epsilon_greedy_route
  outcome_callback: my_router.bandit:update_bandit_scores
  initial_state:
    epsilon: 0.2
    learning_rate: 0.1
    model_scores:
      small: 0.5
      large: 0.5
```

## Adaptive Mode

When `adaptive: true` is set:

1. The router processes requests one document at a time
2. After each document, `outcome_callback` is called with results
3. The callback can update `state` to influence future routing

This enables online learning algorithms to adapt during a single evaluation run.

## Calling Multiple Models from Router

If your router needs to evaluate multiple models (e.g., for dataset creation or comparison), you can call models directly:

```python
class DatasetCreatingRouter:
    def __init__(self, models):
        self._models = models
    
    @property
    def models(self):
        return self._models
    
    def route(self, request, context, state):
        # Get the selected model
        selected = self._select_model(request)
        
        # Optionally call all models for dataset creation
        if state.get("capture_all", False):
            for model_id, model in self._models.items():
                response = model.generate_until([request])[0]
                self._store_response(request, model_id, response)
        
        # Return the selected model for evaluation
        return selected
    
    def update(self, event, state):
        pass
```

## Class-Based Routers (Recommended)

For complex routers with stateful models (e.g., neural classifiers), use the class-based pattern with `RouterLM.from_router()`.

### Required Interface

Your router class must implement:

```python
class MyRouter:
    @property
    def models(self) -> dict[str, LM]:
        """Return dict of model_id -> LM instance."""
        return self._models
    
    def route(self, request: Instance, context: RoutingContext, state: dict) -> str | RoutingDecision:
        """Make routing decision."""
        ...
    
    def update(self, event: OutcomeEvent, state: dict) -> None:
        """Process outcome feedback."""
        ...
```

### Optional Methods

```python
def get_state(self) -> dict:
    """Return serializable state for checkpointing."""
    ...

def set_state(self, state: dict) -> None:
    """Restore state from checkpoint."""
    ...

def seed(self, seed: int) -> None:
    """Set random seed for reproducibility."""
    ...
```

### Complete Example

```python
# my_router/classifier_router.py
import torch
import torch.nn as nn
from lm_eval.api.model import LM
from lm_eval.api.router import RoutingContext, RoutingDecision, OutcomeEvent
from lm_eval.models.router import RouterLM

class ClassifierRouter:
    """Routes based on a trained classifier."""
    
    def __init__(self, models: dict[str, LM], classifier_path: str):
        self._models = models
        self.classifier = self._load_classifier(classifier_path)
        self.outcomes_received = 0
    
    def _load_classifier(self, path: str) -> nn.Module:
        classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, len(self._models))
        )
        classifier.load_state_dict(torch.load(path))
        classifier.eval()
        return classifier
    
    @property
    def models(self) -> dict[str, LM]:
        return self._models
    
    def route(self, request, context, state):
        """Route based on classifier prediction."""
        # Get embedding from context (your logic here)
        embedding = self._get_embedding(context)
        
        with torch.no_grad():
            logits = self.classifier(embedding)
            model_idx = logits.argmax().item()
        
        model_ids = list(self._models.keys())
        return model_ids[model_idx]
    
    def update(self, event, state):
        """Track outcomes (can be used for fine-tuning)."""
        self.outcomes_received += 1
    
    def get_state(self):
        return {"outcomes_received": self.outcomes_received}
    
    def set_state(self, state):
        self.outcomes_received = state.get("outcomes_received", 0)

# Usage
from lm_eval.models.huggingface import HFLM

models = {
    "small": HFLM(pretrained="gpt2"),
    "large": HFLM(pretrained="gpt2-medium"),
}

router = ClassifierRouter(models, classifier_path="classifier.pt")
router_lm = RouterLM.from_router(router)

# Run evaluation
# lm-eval will use router.route() and router.update() automatically
```

### Loading Models from YAML

For complex setups, load models from a YAML config:

```yaml
# router_config.yaml
models:
  small:
    type: hf
    pretrained: gpt2
    metadata:
      params_billion: 0.124
  large:
    type: hf
    pretrained: gpt2-medium
    metadata:
      params_billion: 0.355
```

```python
from lm_eval.models.router import RouterLM
from lm_eval.api.router import RoutingContext, RoutingDecision, OutcomeEvent

class MyRouter:
    def __init__(self):
        self._models = {}
    
    @property
    def models(self):
        return self._models
    
    def route(self, request, context, state):
        return "small"
    
    def update(self, event, state):
        pass

# Create empty RouterLM to parse config, then attach to your router
base = RouterLM(config_path="router_config.yaml")
router = MyRouter()
router._models = base.models
router_lm = RouterLM.from_router(router)
```

## Complete Example: MESS+ Style Router

```python
# my_router/mess_plus.py
import random
from lm_eval.api.router import RoutingDecision

def mess_route(request, context, state):
    """Model Ensemble with Smart Selection (MESS+) routing."""
    models = state.get("models", ["small", "large"])
    scores = state.get("model_scores", {m: 0.5 for m in models})
    epsilon = state.get("epsilon", 0.1)
    
    # Get prompt features
    prompt = context.arguments[0] if context.arguments else ""
    features = {
        "length": len(prompt),
        "task": context.task_name,
    }
    state["last_features"] = features
    
    # Epsilon-greedy exploration
    if random.random() < epsilon:
        selected = random.choice(models)
    else:
        selected = max(scores, key=scores.get)
    
    return RoutingDecision(
        model=selected,
        metadata={"features": features}
    )

def mess_update(event, state):
    """Update MESS+ model scores."""
    scores = state.setdefault("model_scores", {})
    alpha = state.get("learning_rate", 0.1)
    
    current = scores.get(event.model, 0.5)
    reward = 1.0 if event.correct else 0.0
    scores[event.model] = (1 - alpha) * current + alpha * reward
    
    # Log for analysis
    history = state.setdefault("history", [])
    history.append({
        "task": event.task_name,
        "doc_id": event.doc_id,
        "model": event.model,
        "correct": event.correct,
        "latency_ms": event.latency_ms,
    })
```

Configuration:
```yaml
models:
  small:
    type: hf
    pretrained: gpt2
    metadata:
      params_billion: 0.124
  large:
    type: hf
    pretrained: gpt2-medium
    metadata:
      params_billion: 0.355

routing:
  adaptive: true
  routing_callback: my_router.mess_plus:mess_route
  outcome_callback: my_router.mess_plus:mess_update
  initial_state:
    models: [small, large]
    epsilon: 0.15
    learning_rate: 0.1
    model_scores:
      small: 0.5
      large: 0.5
```

## API Reference

### RoutingContext

```python
@dataclass
class RoutingContext:
    request_type: str          # "loglikelihood", "generate_until", etc.
    task_name: str | None      # Task name
    doc_id: int | None         # Document index
    doc: dict[str, Any]        # Document content
    arguments: tuple           # Request arguments
    idx: int                   # Instance index
    batch_context: dict        # Additional batch context
```

### RoutingDecision

```python
@dataclass
class RoutingDecision:
    model: str                      # Model to route to
    metadata: dict[str, Any]        # Custom metadata
```

### OutcomeEvent

```python
@dataclass
class OutcomeEvent:
    request: Instance               # Original request
    task_name: str                  # Task name
    doc_id: int                     # Document index
    doc: dict[str, Any]             # Document content
    model: str                      # Model ID that was used
    response: Any                   # Model's response
    metrics: dict[str, float]       # Evaluation metrics
    correct: bool                   # Whether correct
    routing_metadata: dict[str, Any]  # Routing decision metadata
    latency_ms: float               # Latency in milliseconds
    energy_joules: float            # Energy in joules (if monitored)
```

## Troubleshooting

### Model not found error
Ensure model IDs returned by your callback match the keys in `models`.

### Callback not loading
Verify the Python path format: `module.submodule:function_name`. The module must be importable.

### Adaptive mode not working
Ensure both `adaptive: true` and an `outcome_callback` are configured.
