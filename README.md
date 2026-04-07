---
title: Bpo Env Environment Server
emoji: 🎹
colorFrom: pink
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

openenv build
docker run -p 8000:8000 bpo_env_env:latest
APP_ENV=test python inference.py

--task {order_status,damaged_product,escalation}
python3 run_scenarios.py --url http://localhost:8000 --task order_status

# Bpo Env Environment

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

## Quick Start

The simplest way to use the Bpo Env environment is through the `BpoEnv` class:

```python
from bpo_env import BpoAction, BpoEnv

try:
    # Create environment from Docker image
    bpo_envenv = BpoEnv.from_docker_image("bpo_env-env:latest")

    # Reset
    result = bpo_envenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = bpo_envenv.step(BpoAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    bpo_envenv.close()
```

That's it! The `BpoEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t bpo_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**BpoAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**BpoObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Bpo Env environment server running, you can connect directly:

```python
from bpo_env import BpoEnv

# Connect to existing server
bpo_envenv = BpoEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = bpo_envenv.reset()
result = bpo_envenv.step(BpoAction(message="Hello!"))
```

Note: When connecting to an existing server, `bpo_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from bpo_env import BpoAction, BpoEnv

# Connect with context manager (auto-connects and closes)
with BpoEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(BpoAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    BpoEnvironment,  # Pass class, not instance
    BpoAction,
    BpoObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from bpo_env import BpoAction, BpoEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with BpoEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(BpoAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/bpo_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
bpo_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # BpoEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── bpo_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```


# 🚀 Multi-Task BPO RL Environment

### 🧠 A Structured Evaluation & Training Platform for Customer Support Agents

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![RL Environment](https://img.shields.io/badge/RL-Environment-orange)
![Status](https://img.shields.io/badge/Status-Stable-success)

---

## 🧠 Overview

This project is a **multi-task reinforcement learning (RL) environment** designed to **simulate and evaluate customer support agent behavior** across real-world scenarios.

Unlike traditional chatbot systems that only generate responses, this environment focuses on:

> 🎯 **Evaluating whether an agent behaves correctly — not just what it says**

---

## 🔥 Key Features

* 🧩 **Multi-Task Support**

  * 📦 Order Status
  * 📉 Damaged Product Handling
  * 🚨 Escalation Management

* ⚙️ **State-Based Conversation Engine**

  * Task-specific state machines
  * Dynamic stage transitions (e.g., inquiry → resolution → closure)

* 🎯 **Three-Layer Evaluation System**

  * **Reward** → Step-by-step behavior quality
  * **Rule Score** → Progress & intent completion
  * **Grader Score** → Final task correctness

* 🧠 **Intent-Aware Understanding**

  * Detects:

    * empathy
    * escalation
    * refund
    * replacement
    * tracking info

* 🚫 **Anti-Cheating Mechanisms**

  * Prevents skipping required steps
  * Enforces task-specific completion rules

* 🔄 **Recovery Handling**

  * Supports agents that recover after poor initial responses

---

## 🏗️ System Architecture

```
User Input → Agent Response → Environment Step Engine
                           ↓
                  Intent Extraction Layer
                           ↓
              State + Stage Transition Logic
                           ↓
        Reward + Rule Score + Grader Evaluation
                           ↓
                   Structured Feedback Output
```

---

## 🧪 Supported Scenarios

### 📦 Order Status

* Empathy → Tracking Info → Delivery → Closure

### 📉 Damaged Product

* Apology → Diagnosis → Replacement/Refund → Closure

### 🚨 Escalation

* De-escalation → Manager Escalation → Refund → Closure

---

## ⚙️ How It Works

1. Customer sends a query
2. Agent generates a response
3. Environment:

   * Extracts intents
   * Updates stage
   * Applies reward logic
4. Outputs structured evaluation:

   * Reward
   * Rule Score
   * Grader Score
   * Stage
   * Mood

---

## 📊 Evaluation Metrics

### 🟢 Reward

* Measures **behavior quality at each step**
* Penalizes:

  * irrelevant responses
  * repetition
  * stalling
* Rewards:

  * correct actions
  * proper sequencing
  * recovery

---

### 🔵 Rule Score

* Tracks **progress toward completion**
* Stage-aware + intent-aware

---

### 🟣 Grader Score

* Final evaluation of:

  * task completion
  * required intents
  * proper closure

---

## 🛡️ Robustness Features

* ✔️ Handles incomplete responses
* ✔️ Penalizes irrelevant behavior
* ✔️ Detects stalling patterns
* ✔️ Supports recovery flows
* ✔️ Prevents reward exploitation

---

## 🚀 Example Flow

```
Customer: My product arrived damaged.

Agent:
1. Apologizes → (Reward ↑)
2. Asks for details → (Stage progression)
3. Offers replacement → (High reward)
4. Confirms resolution → (Grader success)
```

---

## 🧰 Tech Stack

* 🐍 Python
* ⚡ FastAPI
* 🧠 OpenEnv (RL Environment Framework)
* 🔍 Rule-Based Intent Detection
* 🎯 Reinforcement Learning Style Reward System

---

## 💡 Use Cases

* Training RL-based customer support agents
* Benchmarking conversational AI systems
* Simulating real-world BPO workflows
* Evaluating agent reliability and correctness

---

## ⚡ Future Improvements

* 🔌 LLM-based semantic evaluation layer
* 📈 Full RL training loop integration
* 📊 Visualization dashboard
* 🌍 Expanded scenario coverage

---

## 🎯 Why This Project Matters

Most AI systems focus on:

> ❌ Generating responses

This system focuses on:

> ✅ **Evaluating correct behavior in real-world workflows**

---

## 👥 Team — *Skill Hive*

* 👤 Raja Guru R
* 👤 Prasanna S
* 👤 Hariharan P

---

## 🏁 Conclusion

This project demonstrates how to build a:

> 🧠 **Structured, scalable, and intelligent evaluation environment**

for real-world conversational AI systems.

---

## ⭐ Final Thought

> “Most systems generate answers.
> This system evaluates whether those answers are *correct*.”

