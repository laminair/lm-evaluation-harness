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
  large:
    type: hf
    pretrained: gpt2-medium

routing:
  primary_model: small
  shadow_mode: none
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
  
  gpt_large:
    type: hf
    pretrained: gpt2-medium
    dtype: float32
  
  vllm_model:
    type: vllm
    pretrained: meta-llama/Llama-2-7b-hf
    tensor_parallel_size: 2
```

### Routing Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `primary_model` | string | first model | Default model to route requests to |
| `shadow_mode` | string | `"none"` | Shadow routing mode: `"none"`, `"all"`, or `"sampled"` |
| `shadow_sample_rate` | float | `0.5` | Fraction of models to sample when `shadow_mode: sampled` |
| `adaptive` | bool | `false` | Enable online learning with outcome feedback |
| `routing_callback` | string | none | Python path to routing function |
| `outcome_callback` | string | none | Python path to outcome feedback function |
| `initial_state` | dict | `{}` | Initial state passed to callbacks |

### Shadow Routing Modes

- **`none`**: Only the primary model executes each request
- **`all`**: All models execute each request (for training data collection)
- **`sampled`**: Random subset of models execute each request

## Writing Routing Callbacks

A routing callback decides which model(s) should handle each request.

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

1. **String**: Model ID to route to (no shadow evaluation)
   ```python
   return "large_model"
   ```

2. **RoutingDecision**: Primary model with optional shadow models
   ```python
   from lm_eval.api.router import RoutingDecision
   return RoutingDecision(
       primary_model="large_model",
       shadow_models=["small_model"],
       metadata={"reason": "complex_query"}
   )
   ```

3. **Dict**: Shorthand for RoutingDecision
   ```python
   return {
       "primary": "large_model",
       "shadow": ["small_model"],
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
  primary_model: small_model
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
  primary_model: small_model
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
| `primary_model` | str | Model that was used for metrics |
| `shadow_models` | list[str] | Shadow models that were evaluated |
| `all_responses` | dict | Model ID → response for all models |
| `primary_metrics` | dict | Metrics for primary model |
| `primary_correct` | bool | Whether primary model was correct |
| `all_metrics` | dict | Model ID → metrics dict for all models |
| `all_correct` | dict | Model ID → bool for all models |
| `routing_metadata` | dict | Metadata from routing decision |

### Example: Update Bandit Scores

```python
# my_router/bandit.py

def update_bandit_scores(event, state):
    """Update model scores using exponential moving average."""
    scores = state.setdefault("model_scores", {})
    alpha = state.get("learning_rate", 0.1)
    
    for model_id, correct in event.all_correct.items():
        current = scores.get(model_id, 0.5)
        reward = 1.0 if correct else 0.0
        scores[model_id] = (1 - alpha) * current + alpha * reward
    
    # Decay epsilon over time
    epsilon = state.get("epsilon", 0.1)
    state["epsilon"] = max(0.01, epsilon * 0.999)
```

Configuration for adaptive routing:
```yaml
routing:
  primary_model: small_model
  adaptive: true
  shadow_mode: all
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
4. Primary model metrics are used for final evaluation

This enables online learning algorithms to adapt during a single evaluation run.

## Shadow Routing for Training

Use shadow routing to collect data for training a router model:

```yaml
routing:
  primary_model: small_model
  shadow_mode: all
```

All models will evaluate every request. The `all_metrics` and `all_correct` fields in `OutcomeEvent` contain results for all models, which you can log for later analysis.

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
        primary = random.choice(models)
    else:
        primary = max(scores, key=scores.get)
    
    # Always evaluate shadows for learning
    shadows = [m for m in models if m != primary]
    
    return RoutingDecision(
        primary_model=primary,
        shadow_models=shadows,
        metadata={"features": features}
    )

def mess_update(event, state):
    """Update MESS+ model scores."""
    scores = state.setdefault("model_scores", {})
    alpha = state.get("learning_rate", 0.1)
    
    for model_id, correct in event.all_correct.items():
        current = scores.get(model_id, 0.5)
        reward = 1.0 if correct else 0.0
        scores[model_id] = (1 - alpha) * current + alpha * reward
    
    # Log for analysis
    history = state.setdefault("history", [])
    history.append({
        "task": event.task_name,
        "doc_id": event.doc_id,
        "primary": event.primary_model,
        "correct": event.primary_correct,
        "all_correct": event.all_correct,
    })
```

Configuration:
```yaml
models:
  small:
    type: hf
    pretrained: gpt2
  large:
    type: hf
    pretrained: gpt2-medium

routing:
  primary_model: small
  adaptive: true
  shadow_mode: all
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
    primary_model: str                 # Model to use for metrics
    shadow_models: list[str]           # Models to evaluate for learning
    metadata: dict[str, Any]           # Custom metadata
```

### OutcomeEvent

```python
@dataclass
class OutcomeEvent:
    request: Instance                  # Original request
    task_name: str                     # Task name
    doc_id: int                        # Document index
    doc: dict[str, Any]                # Document content
    primary_model: str                 # Primary model ID
    shadow_models: list[str]           # Shadow model IDs
    all_responses: dict[str, Any]      # Model ID → response
    primary_metrics: dict[str, float]  # Primary model metrics
    primary_correct: bool              # Primary model correctness
    all_metrics: dict[str, dict]       # Model ID → metrics
    all_correct: dict[str, bool]       # Model ID → correctness
    routing_metadata: dict[str, Any]   # Routing decision metadata
```

## Troubleshooting

### Model not found error
Ensure model IDs in `routing.primary_model` and callback returns match the keys in `models`.

### Callback not loading
Verify the Python path format: `module.submodule:function_name`. The module must be importable.

### Shadow models not executing
Check that `shadow_mode` is not set to `"none"`, or that your callback returns shadow models in the `RoutingDecision`.

### Adaptive mode not working
Ensure both `adaptive: true` and an `outcome_callback` are configured.
