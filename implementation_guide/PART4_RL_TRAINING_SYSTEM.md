# Part 4: RL Training System

**Deep-Dive Documentation for BiG-RAG Framework**

**Note:** This guide focuses on BiG-RAG's RL training integration. BiG-RAG uses GRPO (Group-Relative Policy Optimization) via the external VERL library. This guide covers reward computation, training configuration, and usage from BiG-RAG's perspective.

---

## Table of Contents

1. [Conceptual Overview](#1-conceptual-overview)
2. [Implementation Details](#2-implementation-details)
3. [Configuration Reference](#3-configuration-reference)
4. [Usage Examples](#4-usage-examples)
5. [Troubleshooting](#5-troubleshooting)
6. [API Reference](#6-api-reference)
7. [Performance Analysis](#7-performance-analysis)
8. [Testing Guide](#8-testing-guide)

---

## 1. Conceptual Overview

### What Problem Does This Solve?

**Problem:** LLMs don't naturally know WHEN or HOW to use retrieval tools:
- **No incentive**: Standard training doesn't reward effective tool use
- **Random queries**: Without guidance, models query arbitrarily
- **Inefficient**: May retrieve irrelevant information or skip necessary queries
- **No optimization**: Can't learn from query success/failure

**BiG-RAG Solution:** Use **Reinforcement Learning** to optimize tool use:
```
Standard Fine-Tuning:
  Loss = CrossEntropy(model_output, ground_truth)
  Problem: No signal for tool usage quality

RL Training (GRPO):
  Reward = Format_Reward + Answer_Quality_Reward
  - Format_Reward: Correct tool call structure
  - Answer_Quality: EM/F1 score vs. ground truth

  Policy Update: Increase probability of high-reward trajectories
```

### Why GRPO (Group-Relative Policy Optimization)?

**Comparison with Other RL Algorithms:**

| Algorithm | Baseline | Sample Efficiency | Stability | Complexity |
|-----------|----------|-------------------|-----------|------------|
| **REINFORCE** | None | Low | Unstable | Low |
| **PPO** | Critic network | Medium | Stable | High |
| **GRPO** | **Group mean** | **High** | **Very Stable** | **Medium** |
| **DPO** | Reference policy | High | Stable | Low |

**GRPO Advantages:**

1. **No Critic Network**: Uses group statistics as baseline (simpler than PPO)
2. **Group-Relative Rewards**: Normalizes within batches (stable training)
3. **Sample Efficient**: Learns from multiple rollouts per prompt
4. **KL Regularization**: Stays close to reference policy (prevents forgetting)

**GRPO Formula:**
```
For each prompt, generate n_repeat responses (e.g., 4 responses)

Group advantage:
  A_i = (R_i - mean(R_group)) / (std(R_group) + ε)

Policy gradient:
  ∇J = E[A_i × ∇ log π(a_i | s_i)]

Effect: Responses with above-average rewards are reinforced
```

### High-Level Training Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL TRAINING PIPELINE                          │
└─────────────────────────────────────────────────────────────────┘

Input: Training Dataset (Parquet)
  [{prompt, question, answers}, ...]

   ↓

┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: ROLLOUT (Generation with Tools)                       │
├─────────────────────────────────────────────────────────────────┤
│  Component: Actor + ToolGenerationManager                       │
│                                                                  │
│  For each training prompt:                                      │
│    1. Generate n_repeat responses (e.g., 4)                    │
│    2. Each response uses tool-augmented generation             │
│    3. Tools retrieve knowledge during generation               │
│    4. Collect full trajectories:                               │
│       - LLM responses                                           │
│       - Tool calls                                              │
│       - Tool responses                                          │
│       - Final answers                                           │
│                                                                  │
│  Output: n_repeat trajectories per prompt                       │
└─────────────────────────────────────────────────────────────────┘

   ↓

┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: REWARD COMPUTATION                                     │
├─────────────────────────────────────────────────────────────────┤
│  Component: RewardManager                                        │
│                                                                  │
│  For each trajectory:                                            │
│    1. Extract answer from <answer>...</answer> tags            │
│    2. Compute format reward:                                    │
│       - Has <think> tags? (+0.5)                               │
│       - Has <query> tags? (+0.5)                               │
│       - Has <answer> tags? (+0.5)                              │
│       - Format complete? (Sum to 1.0)                          │
│    3. Compute answer reward:                                    │
│       - Exact Match (EM): 1.0 or 0.0                          │
│       - F1 Score: Token overlap (0.0-1.0)                     │
│    4. Combined reward:                                          │
│       IF format < 1.0:                                         │
│         reward = format_score - 1.0  (penalty)                │
│       ELSE:                                                     │
│         reward = 1.0 + answer_score - 1.0 = answer_score      │
│                                                                  │
│  Output: Reward for each trajectory                             │
└─────────────────────────────────────────────────────────────────┘

   ↓

┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: ADVANTAGE ESTIMATION (GRPO)                           │
├─────────────────────────────────────────────────────────────────┤
│  Component: GRPO Advantage Estimator                            │
│                                                                  │
│  For each group (n_repeat responses to same prompt):           │
│    1. Calculate group statistics:                               │
│       mean_reward = mean(R_1, R_2, ..., R_n)                  │
│       std_reward = std(R_1, R_2, ..., R_n)                    │
│                                                                  │
│    2. Compute advantages:                                       │
│       A_i = (R_i - mean_reward) / (std_reward + 1e-8)         │
│                                                                  │
│  Effect:                                                         │
│    - Above-average responses: A_i > 0 (reinforced)            │
│    - Below-average responses: A_i < 0 (discouraged)           │
│    - Normalized scale prevents extreme updates                 │
└─────────────────────────────────────────────────────────────────┘

   ↓

┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: POLICY UPDATE                                          │
├─────────────────────────────────────────────────────────────────┤
│  Component: Actor Policy Gradient                               │
│                                                                  │
│  Policy Gradient with KL Regularization:                        │
│    L_policy = -E[A_i × log π_θ(a_i | s_i)]                   │
│    L_KL = KL(π_θ || π_ref)                                     │
│    L_total = L_policy + β × L_KL                              │
│                                                                  │
│  Where:                                                          │
│    π_θ = Current policy (being trained)                        │
│    π_ref = Reference policy (frozen initial model)             │
│    β = KL coefficient (default 0.001)                          │
│                                                                  │
│  Update:                                                         │
│    θ = θ - α × ∇L_total                                        │
│                                                                  │
│  Loss Masking:                                                   │
│    - Gradients only on LLM-generated tokens                    │
│    - NO gradients on tool-provided <knowledge> content        │
│    - Prevents memorizing retrieval results                     │
└─────────────────────────────────────────────────────────────────┘

   ↓

┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5: CHECKPOINT & EVALUATION                                │
├─────────────────────────────────────────────────────────────────┤
│  • Save model checkpoint every N steps                          │
│  • Evaluate on validation set                                   │
│  • Log metrics to Weights & Biases:                            │
│    - Mean reward                                                │
│    - Reward std                                                 │
│    - EM score                                                   │
│    - F1 score                                                   │
│    - KL divergence                                              │
│    - Policy loss                                                │
│    - Learning rate                                              │
└─────────────────────────────────────────────────────────────────┘

   ↓

Output: Trained Model Checkpoint
  Model learned to:
  ✓ Generate valid tool calls
  ✓ Query relevant information
  ✓ Provide accurate answers
```

---

## 2. Implementation Details

### Reward Computation Algorithm

**File:** `verl/utils/reward_score/qa_em_and_format.py`

```python
ALGORITHM: Compute_Reward
INPUT: response: str, ground_truth: List[str]
OUTPUT: reward: float

PROCEDURE compute_reward(response, ground_truth):
    # Step 1: Compute format reward
    format_reward = compute_format_reward(response)

    # Step 2: Compute answer reward (if format complete)
    IF format_reward >= 1.0:
        answer_reward = compute_answer_reward(response, ground_truth)
    ELSE:
        answer_reward = 0.0

    # Step 3: Combined reward
    IF format_reward < 1.0:
        # Incomplete format → penalty
        reward = format_reward - 1.0
    ELSE:
        # Complete format → answer score
        reward = answer_reward

    RETURN reward

END PROCEDURE


FUNCTION compute_format_reward(response: str) -> float:
    """
    Check if response has correct structure

    Expected format:
      <think>...</think>
      <query>...</query>
      <knowledge>...</knowledge>
      <answer>...</answer>
    """

    score = 0.0

    # Check for think tags (0.5 points)
    IF "<think>" IN response AND "</think>" IN response:
        score += 0.5

    # Check for query tags (0.5 points)
    IF "<query>" IN response AND "</query>" IN response:
        score += 0.5

    # Alternative: answer tags without query (direct answer)
    ELIF "<answer>" IN response AND "</answer>" IN response:
        score += 0.5

    # Check for answer tags (1.0 points total if all present)
    IF "<answer>" IN response AND "</answer>" IN response:
        score = 1.0

    RETURN score

END FUNCTION


FUNCTION compute_answer_reward(response: str, ground_truth: List[str]) -> float:
    """
    Compute answer quality score

    Combines:
    - Exact Match (EM): 1.0 or 0.0
    - F1 Score: Token overlap (0.0-1.0)
    """

    # Extract answer
    answer = extract_answer(response)  # Between <answer>...</answer>

    # Normalize
    answer_normalized = normalize_text(answer)
    ground_truth_normalized = [normalize_text(gt) for gt IN ground_truth]

    # Compute EM
    em_score = 0.0
    FOR gt IN ground_truth_normalized:
        IF answer_normalized == gt:
            em_score = 1.0
            BREAK

    # Compute F1
    f1_scores = []
    FOR gt IN ground_truth_normalized:
        f1 = compute_f1(answer_normalized, gt)
        f1_scores.append(f1)

    f1_score = max(f1_scores) IF f1_scores ELSE 0.0

    # Weighted combination
    answer_reward = 0.5 × em_score + 0.5 × f1_score

    RETURN answer_reward

END FUNCTION


FUNCTION compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score

    F1 = 2 × (precision × recall) / (precision + recall)
    """

    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    IF len(pred_tokens) == 0 OR len(gt_tokens) == 0:
        RETURN 0.0

    # Common tokens
    common = set(pred_tokens) & set(gt_tokens)

    # Precision and recall
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    IF precision + recall == 0:
        RETURN 0.0

    f1 = 2 × (precision × recall) / (precision + recall)

    RETURN f1

END FUNCTION
```

### GRPO Advantage Computation

**Conceptual Implementation:**

```python
ALGORITHM: GRPO_Advantage_Estimation
INPUT: rewards: List[float], group_ids: List[int]
OUTPUT: advantages: List[float]

PROCEDURE compute_grpo_advantages(rewards, group_ids):
    """
    Compute group-relative advantages

    Each group corresponds to n_repeat responses for same prompt
    """

    # Group rewards by prompt ID
    groups = {}  # {group_id: [reward_1, reward_2, ...]}

    FOR i, group_id IN enumerate(group_ids):
        IF group_id NOT IN groups:
            groups[group_id] = []
        groups[group_id].append((i, rewards[i]))

    # Compute advantages
    advantages = [0.0] * len(rewards)

    FOR group_id, group_rewards IN groups.items():
        # Extract rewards for this group
        indices = [idx for idx, _ IN group_rewards]
        group_r = [r for _, r IN group_rewards]

        # Group statistics
        mean_r = mean(group_r)
        std_r = std(group_r)

        # Normalize
        FOR idx, r IN zip(indices, group_r):
            IF std_r > 0:
                advantages[idx] = (r - mean_r) / (std_r + 1e-8)
            ELSE:
                advantages[idx] = 0.0  # All same reward

    RETURN advantages

END PROCEDURE
```

**Example:**
```
Prompt: "What is Paris?"

Rollouts (n_repeat=4):
  Response 1: <query>search Paris</query> <answer>Capital of France</answer>
    Reward: 1.0 (correct)
  Response 2: I don't know.
    Reward: 0.0 (no answer)
  Response 3: <query>invalid json</query>
    Reward: -0.5 (format error)
  Response 4: <answer>A city</answer>
    Reward: 0.3 (partial F1)

Group statistics:
  mean_reward = (1.0 + 0.0 + (-0.5) + 0.3) / 4 = 0.2
  std_reward = 0.56

Advantages:
  A_1 = (1.0 - 0.2) / 0.56 = 1.43  (strong positive)
  A_2 = (0.0 - 0.2) / 0.56 = -0.36 (negative)
  A_3 = (-0.5 - 0.2) / 0.56 = -1.25 (strong negative)
  A_4 = (0.3 - 0.2) / 0.56 = 0.18  (weak positive)

Effect: Response 1 strongly reinforced, Response 3 strongly discouraged
```

---

## 3. Configuration Reference

### Training Configuration

**Main Config File:** `verl/trainer/config/ppo_trainer.yaml`

**Key Parameters:**

```yaml
# Data configuration
data:
  train_files: ['datasets/2WikiMultiHopQA/processed/train.parquet']
  val_files: ['datasets/2WikiMultiHopQA/processed/dev.parquet']
  train_batch_size: 128       # Global batch size across all GPUs
  val_batch_size: 64
  max_prompt_length: 4096     # Maximum input tokens
  max_response_length: 4096   # Maximum generation tokens

# Model configuration
actor_rollout_ref:
  model:
    path: "Qwen/Qwen2.5-3B-Instruct"  # Model path or HF ID

  # Actor (policy being trained)
  actor:
    optim:
      lr: 5e-7                 # Learning rate
      weight_decay: 0.01
      betas: [0.9, 0.95]

    ppo_mini_batch_size: 32    # Mini-batch for gradient updates
    ppo_micro_batch_size_per_gpu: 1

  # Rollout (generation)
  rollout:
    name: 'vllm'               # Use vLLM for fast generation
    gpu_memory_utilization: 0.5  # Fraction of GPU for vLLM
    tensor_model_parallel_size: 4  # GPUs for model parallelism
    n_repeat: 4                # Rollouts per prompt (for GRPO)

  # Reference policy (frozen)
  ref:
    log_prob_micro_batch_size_per_gpu: 4
    fsdp_config: ...           # FSDP configuration

# Algorithm configuration
algorithm:
  adv_estimator: 'grpo'        # Use GRPO
  gamma: 1.0                   # Discount factor (1.0 = no discounting)
  lam: 0.95                    # GAE lambda (unused in GRPO)

  # KL divergence control
  kl_ctrl:
    type: 'fixed'              # or 'adaptive'
    kl_coef: 0.001             # KL penalty coefficient

# Tool configuration
tool:
  env: 'search'                # Tool environment type
  max_turns: 5                 # Maximum tool interactions
  query_start_tag: "<query>"
  query_end_tag: "</query>"

# Trainer configuration
trainer:
  total_epochs: 1              # Training duration
  test_freq: 10                # Evaluate every N steps
  save_freq: 100               # Checkpoint every N steps
  n_gpus_per_node: 4           # GPUs per machine
  nnodes: 1                    # Number of machines
  critic_warmup: 0             # Critic warmup steps (N/A for GRPO)
```

### Runtime Overrides

**Training Scripts:** `run_grpo.sh`

```bash
#!/bin/bash

MODEL_PATH=$1    # e.g., "Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME=$2    # e.g., "qwen3b"
DATASET=$3       # e.g., "2WikiMultiHopQA"

# Environment variables
export VLLM_ATTENTION_BACKEND=XFORMERS
export PROJECT_NAME='BiG-RAG'
export EXPERIMENT_NAME="${MODEL_NAME}_${DATASET}_grpo"

# Launch training with Hydra overrides
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=["datasets/${DATASET}/processed/train.parquet"] \
    data.val_files=["datasets/${DATASET}/processed/dev.parquet"] \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.rollout.n_repeat=4 \
    trainer.total_epochs=1 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    tool.env=search \
    tool.max_turns=5
```

### Key Parameter Tuning

**Learning Rate:**
```yaml
# Conservative (safer, slower)
actor.optim.lr: 1e-7

# Default (balanced)
actor.optim.lr: 5e-7

# Aggressive (faster, less stable)
actor.optim.lr: 1e-6
```

**n_repeat (GRPO Group Size):**
```yaml
# Small group (less stable, faster)
rollout.n_repeat: 2

# Default (balanced)
rollout.n_repeat: 4

# Large group (more stable, slower)
rollout.n_repeat: 8
```

**Trade-off:** Larger n_repeat = better advantage estimation but slower training

**KL Coefficient:**
```yaml
# Weak regularization (more deviation from reference)
kl_ctrl.kl_coef: 0.0001

# Default (balanced)
kl_ctrl.kl_coef: 0.001

# Strong regularization (stays close to reference)
kl_ctrl.kl_coef: 0.01
```

**Trade-off:** Higher KL = safer but may limit learning

---

## 4. Usage Examples

### Basic Training

**Start Training:**

```bash
# 1. Ensure retrieval server is running
python script_api.py --data_source 2WikiMultiHopQA &

# 2. Start Ray cluster
ray start --head

# 3. Launch training
bash run_grpo.sh \
  Qwen/Qwen2.5-3B-Instruct \
  qwen3b \
  2WikiMultiHopQA

# 4. Monitor logs
tail -f training.log

# 5. View metrics in W&B
# Navigate to https://wandb.ai/your-project
```

### Advanced Scenarios

**Scenario 1: Multi-GPU Training**

```bash
# Configure for 8 GPUs
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    trainer.n_gpus_per_node=8 \
    data.train_batch_size=256  # Increase batch size
```

**Scenario 2: Custom Reward Function**

```python
# File: custom_reward.py

def custom_compute_reward(response: str, ground_truth: List[str]) -> float:
    """Custom reward with domain-specific bonuses"""

    # Base reward
    base_reward = compute_reward(response, ground_truth)

    # Bonus for using specific tools
    if "search" in response and "<query>" in response:
        base_reward += 0.1

    # Penalty for too many tool calls
    tool_calls = response.count("<query>")
    if tool_calls > 5:
        base_reward -= 0.05 * (tool_calls - 5)

    return base_reward

# Register custom reward
# (requires modifying trainer code)
```

**Scenario 3: Resuming from Checkpoint**

```bash
# Resume training from saved checkpoint
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.load_checkpoint="./checkpoints/step_1000" \
    trainer.total_epochs=2  # Train for 1 more epoch
```

### Common Patterns

**Pattern 1: Monitoring Training**

```python
# View training metrics
import wandb

run = wandb.init(
    project="BiG-RAG",
    name="qwen3b_2wiki_grpo",
    resume="allow"
)

# Key metrics to watch:
# - reward/mean: Should increase over time
# - reward/std: Should decrease (more consistent)
# - metrics/em: Exact match accuracy
# - metrics/f1: Token overlap score
# - kl_divergence: Should stay < 0.1
# - loss/policy: Should decrease
```

**Pattern 2: Hyperparameter Search**

```python
# Grid search over learning rates
learning_rates = [1e-7, 5e-7, 1e-6]
n_repeats = [2, 4, 8]

for lr in learning_rates:
    for n_repeat in n_repeats:
        exp_name = f"lr{lr}_n{n_repeat}"

        # Launch training
        subprocess.run([
            "python", "-m", "verl.trainer.main_ppo",
            f"actor_rollout_ref.actor.optim.lr={lr}",
            f"actor_rollout_ref.rollout.n_repeat={n_repeat}",
            f"trainer.experiment_name={exp_name}"
        ])
```

---

## 5. Troubleshooting

### Common Issues

#### Issue 1: Training Not Converging

**Symptoms:**
- Reward not increasing
- High loss values
- Random performance

**Solutions:**

```bash
# 1. Check reward distribution
# In W&B: Look at reward/mean and reward/std
# Expected: mean increases, std decreases

# 2. Reduce learning rate
actor.optim.lr=1e-7  # Down from 5e-7

# 3. Increase n_repeat
rollout.n_repeat=8  # Up from 4

# 4. Check KL divergence
# If KL > 0.5: Model diverging from reference
# Solution: Increase kl_coef
```

#### Issue 2: Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

```bash
# 1. Reduce batch size
data.train_batch_size=64  # Down from 128

# 2. Reduce vLLM memory
actor_rollout_ref.rollout.gpu_memory_utilization=0.3

# 3. Increase model parallelism
actor_rollout_ref.rollout.tensor_model_parallel_size=8

# 4. Reduce context lengths
data.max_prompt_length=2048
data.max_response_length=2048
```

#### Issue 3: Retrieval Server Timeout

**Symptoms:**
```
requests.exceptions.Timeout: Connection timeout
Tool execution failed
```

**Solutions:**

```bash
# 1. Check server status
curl http://localhost:8001/health

# 2. Restart server if needed
fuser -k 8001/tcp
python script_api.py --data_source 2WikiMultiHopQA &

# 3. Increase timeout in tool
# Modify agent/tool/tools/search_tool.py
timeout=60  # Increase from 30
```

---

## 6. API Reference

### Reward Functions

```python
def compute_score_format_answer(
    response: str,
    ground_truth: List[str]
) -> float:
    """
    Compute combined format + answer reward

    Args:
        response: LLM-generated response
        ground_truth: List of acceptable answers

    Returns:
        Reward score (typically -1.0 to 1.0)

    Raises:
        None (returns 0.0 on error)
    """

def compute_score_answer_em(
    prediction: str,
    ground_truth: List[str]
) -> float:
    """
    Exact match score

    Returns:
        1.0 if exact match, 0.0 otherwise
    """

def compute_score_answer_f1(
    prediction: str,
    ground_truth: List[str]
) -> float:
    """
    Token-level F1 score

    Returns:
        F1 score (0.0-1.0)
    """
```

### Training Scripts

```bash
# Main training entry point
run_grpo.sh <model_path> <model_name> <dataset>

# Example:
bash run_grpo.sh Qwen/Qwen2.5-3B-Instruct qwen3b 2WikiMultiHopQA
```

---

## 7. Performance Analysis

### Training Time

**Benchmark (Qwen2.5-3B on 4x A100 GPUs):**

```
Dataset: 2WikiMultiHopQA (10,000 training samples)
Batch size: 128
n_repeat: 4
Epochs: 1

Stage                   | Time
------------------------|----------
Data loading            | 2 min
Rollout (generation)    | 45 min
Reward computation      | 3 min
Advantage estimation    | 1 min
Policy update           | 20 min
Checkpointing           | 2 min
------------------------|----------
Total per epoch         | 73 min
```

**Scalability:**
- 2x GPUs: ~1.8x speedup (not linear due to communication)
- 4x GPUs: ~3.2x speedup
- 8x GPUs: ~5.5x speedup

---

## 8. Testing Guide

### Validation

```python
def validate_training_run(checkpoint_dir: str) -> Dict:
    """Validate trained model"""

    model = load_checkpoint(checkpoint_dir)

    # Test on validation set
    val_data = load_parquet("datasets/2WikiMultiHopQA/processed/dev.parquet")

    total = 0
    correct = 0

    for sample in val_data:
        response = model.generate(sample['prompt'])
        reward = compute_reward(response, sample['answers'])

        if reward > 0.5:  # Threshold for "correct"
            correct += 1
        total += 1

    accuracy = correct / total

    return {
        "accuracy": accuracy,
        "checkpoint": checkpoint_dir
    }
```

---

## Summary

This guide covers **RL Training** in BiG-RAG:

- **Conceptual Overview**: GRPO for optimizing tool use
- **Implementation**: Reward computation, advantage estimation
- **Configuration**: Training parameters and tuning
- **Usage**: Basic training to advanced scenarios
- **Troubleshooting**: Common issues and solutions

**Key Takeaways:**

1. **GRPO** uses group-relative advantages (simpler than PPO)
2. **Reward = Format + Answer** quality
3. **n_repeat** controls group size for advantage estimation
4. **KL regularization** prevents forgetting
5. **Loss masking** on tool responses prevents memorization

For deployment, see **Part 6: API Server**.
