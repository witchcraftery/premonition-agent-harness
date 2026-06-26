# Precognitive Probability Agent Harness — Implementation Plan

**Version:** 1.0  
**Status:** Speculative Architecture / Engineering Blueprint  
**Companion Documents:** System Spec (v1), Architecture Overview HTML  
**Based on:** FLARE Planning-Centric Analysis (Wang et al., 2026) + MiroFish Swarm Framework

---

## Executive Summary

The Precognitive Probability Agent Harness (PPAH) is a multi-layer cognitive architecture designed to solve the fundamental myopia problem in LLM agents: the inability to sustain coherent behavior across long planning horizons. Rather than improving any single model's lookahead, PPAH separates concerns across three coordinated layers — a real-time **Presenter** delivering current output, a parallel **Swarm** pre-solving the most probable next events, and a background **World-Model** continuously simulating the possibility space. MiroFish serves as the probability engine driving branch generation and world-state simulation.

This document provides a phase-by-phase developer guide, formal module specifications, API contracts, event schemas, evaluation metrics, and post-implementation refinement strategy.

---

## Table of Contents

1. [Architecture Layers](#architecture-layers)
2. [Phase 0 — Foundation](#phase-0--foundation)
3. [Phase 1 — Core Loop](#phase-1--core-loop)
4. [Phase 2 — Swarm Intelligence](#phase-2--swarm-intelligence)
5. [Phase 3 — World-Model Integration](#phase-3--world-model-integration)
6. [Phase 4 — Readiness & Cache Layer](#phase-4--readiness--cache-layer)
7. [Phase 5 — Evaluation Harness](#phase-5--evaluation-harness)
8. [Phase 6 — MiroFish Integration](#phase-6--mirofish-integration)
9. [API Specification](#api-specification)
10. [Event Schemas](#event-schemas)
11. [Module Contracts](#module-contracts)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Speculative Outcomes & Refinement Roadmap](#speculative-outcomes--refinement-roadmap)

---

## Architecture Layers

The three cognitive layers must be understood before any code is written. Every module in this system belongs to exactly one layer.

```
┌────────────────────────────────────────────────────────────┐
│  LAYER 3: POSSIBILITY SPACE  (World-Model + MiroFish)      │
│  • Simulates external reality                              │
│  • Maintains probability tree of next world-states         │
│  • Runs agent swarms per branch hypothesis                 │
│  • Feeds ranked hypotheses to Layer 2                      │
├────────────────────────────────────────────────────────────┤
│  LAYER 2: NEAR-FUTURE  (Swarm Brain)                       │
│  • Receives top-K branches from Layer 3                    │
│  • Spawns one agent per branch                             │
│  • Pre-generates full responses for each branch            │
│  • Deposits artifacts in the Readiness Cache               │
├────────────────────────────────────────────────────────────┤
│  LAYER 1: NOW  (Presenter / Face Model)                    │
│  • Responds to current confirmed events                    │
│  • Pulls pre-solved responses from cache first             │
│  • Falls back to real-time inference only on cache miss    │
│  • Feeds resolved events back to Layer 3                   │
└────────────────────────────────────────────────────────────┘
         ↕ observation                     ↕ resolution
    [EXTERNAL WORLD / USER / ENVIRONMENT]
```

**The key insight:** Layers 2 and 3 are never seen by the user. They run continuously while Layer 1 speaks. The goal is that by the time the next user event arrives, Layer 2 has already answered it.

---

## Phase 0 — Foundation

**Goal:** Establish infrastructure, project structure, and shared data contracts.  
**Duration:** 1–2 weeks  
**Prerequisites:** Docker, Redis ≥7.0, Python ≥3.11, Node ≥20 (optional for API gateway)

### 0.1 — Repository Structure

```
ppah/
├── core/
│   ├── schemas/          # Shared Pydantic models and JSON schemas
│   ├── bus/              # Event bus abstraction (Redis Streams)
│   └── config/           # Environment and runtime config
├── presenter/            # Layer 1 — Face Model service
├── swarm/                # Layer 2 — Branch agent pool
├── world_model/          # Layer 3 — Probability engine
├── cache/                # Readiness cache service (Redis)
├── mirofish/             # MiroFish integration adapter
├── eval/                 # Evaluation harness
├── api/                  # HTTP gateway (FastAPI)
└── tests/
    ├── unit/
    ├── integration/
    └── sim/              # Simulation replay tests
```

### 0.2 — Infrastructure Setup

**docker-compose.yml (minimal)**
```yaml
version: "3.9"
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: redis-server --appendonly yes

  redis-insight:
    image: redislabs/redisinsight:latest
    ports: ["8001:8001"]

  ppah-api:
    build: ./api
    ports: ["8080:8080"]
    environment:
      - REDIS_URL=redis://redis:6379
      - LLM_BACKEND=ollama  # or openai, anthropic, mlx
    depends_on: [redis]

  ppah-presenter:
    build: ./presenter
    depends_on: [redis]

  ppah-swarm:
    build: ./swarm
    depends_on: [redis]
    deploy:
      replicas: 3  # Swarm workers — scale as needed

  ppah-world-model:
    build: ./world_model
    depends_on: [redis]
```

**Environment variables required:**
```
REDIS_URL=redis://localhost:6379
LLM_BACKEND=ollama|openai|anthropic|mlx
LLM_BASE_URL=http://localhost:11434  # Ollama default
LLM_MODEL_PRESENTER=llama3.2:3b     # Small/fast face model
LLM_MODEL_SWARM=llama3.1:8b         # Reasoning model for branches
LLM_MODEL_WORLD=llama3.1:8b         # World-model simulation
MIROFISH_URL=http://localhost:9000   # MiroFish service
TOP_K_BRANCHES=5                     # Max concurrent branch hypotheses
BRANCH_DEPTH=3                       # Steps ahead to simulate
BRANCH_TTL=30                        # Seconds before branch expires
READINESS_THRESHOLD=0.65             # Min score to serve from cache
```

### 0.3 — Install Core Dependencies

```bash
# Python services
pip install fastapi uvicorn pydantic redis[asyncio] httpx \
            openai anthropic instructor tiktoken \
            numpy scipy pytest pytest-asyncio

# Optional: MLX for local Mac inference
pip install mlx-lm  # Apple Silicon only
```

### 0.4 — Shared Config Module

Create `core/config/settings.py`:
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    llm_backend: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_model_presenter: str = "llama3.2:3b"
    llm_model_swarm: str = "llama3.1:8b"
    llm_model_world: str = "llama3.1:8b"
    mirofish_url: str = "http://localhost:9000"
    top_k_branches: int = 5
    branch_depth: int = 3
    branch_ttl: int = 30
    readiness_threshold: float = 0.65

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Phase 1 — Core Loop

**Goal:** Get the Presenter (Layer 1) running as a functioning LLM agent with event observation, response delivery, and observation feedback.  
**Duration:** 1 week  
**Deliverable:** Working single-agent loop that processes events and feeds observations back to a log.

### 1.1 — Event Bus (Redis Streams)

All inter-layer communication uses Redis Streams. There are four primary streams:

| Stream Key | Direction | Description |
|---|---|---|
| `ppah:events:inbound` | External → Layer 1 | Raw user/environment events |
| `ppah:events:resolved` | Layer 1 → Layer 3 | Confirmed reality — what actually happened |
| `ppah:branches:ranked` | Layer 3 → Layer 2 | Ranked hypotheses for next events |
| `ppah:artifacts:ready` | Layer 2 → Layer 1 | Pre-solved response artifacts |

Create `core/bus/streams.py`:
```python
import redis.asyncio as redis
import json
from core.config.settings import settings

class EventBus:
    def __init__(self):
        self.client = redis.from_url(settings.redis_url)

    async def publish(self, stream: str, payload: dict) -> str:
        return await self.client.xadd(stream, {"data": json.dumps(payload)})

    async def consume(self, stream: str, group: str, consumer: str,
                      count: int = 10, block_ms: int = 100):
        try:
            await self.client.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception:
            pass  # Group already exists
        messages = await self.client.xreadgroup(
            group, consumer, {stream: ">"}, count=count, block=block_ms
        )
        return messages

    async def ack(self, stream: str, group: str, message_id: str):
        await self.client.xack(stream, group, message_id)
```

### 1.2 — Presenter Service (Layer 1)

Create `presenter/presenter.py`:
```python
import asyncio
import json
from core.bus.streams import EventBus
from core.schemas.events import InboundEvent, ResolvedEvent
from cache.readiness_cache import ReadinessCache
from llm.llm_client import LLMClient

class Presenter:
    def __init__(self):
        self.bus = EventBus()
        self.cache = ReadinessCache()
        self.llm = LLMClient(model="presenter")
        self.session_id = None

    async def run(self):
        print("[Presenter] Starting observation loop...")
        while True:
            messages = await self.bus.consume(
                stream="ppah:events:inbound",
                group="presenter-group",
                consumer="presenter-1"
            )
            for stream, msgs in (messages or []):
                for msg_id, fields in msgs:
                    event_data = json.loads(fields[b"data"])
                    await self.handle_event(event_data, msg_id, stream)

    async def handle_event(self, event_data: dict, msg_id: bytes, stream: bytes):
        event = InboundEvent(**event_data)

        # 1. Try cache first
        artifact = await self.cache.lookup(
            session_id=event.session_id,
            event_hash=event.content_hash
        )

        if artifact and artifact.readiness_score >= 0.65:
            response = artifact.response_text
            source = "cache_hit"
        else:
            # 2. Fall back to real-time inference
            response = await self.llm.complete(
                system="You are a concise, knowledgeable assistant.",
                user=event.content,
                context=event.context
            )
            source = "live_inference"

        # 3. Deliver response
        await self.deliver(response, event, source)

        # 4. Feed resolved event back to world-model
        resolved = ResolvedEvent(
            session_id=event.session_id,
            original_event=event,
            actual_content=event.content,
            response_delivered=response,
            source=source
        )
        await self.bus.publish("ppah:events:resolved", resolved.dict())
        await self.bus.ack(str(stream, "utf-8"), "presenter-group", str(msg_id, "utf-8"))

    async def deliver(self, response: str, event: InboundEvent, source: str):
        print(f"[Presenter] [{source}] → {response[:80]}...")
```

### 1.3 — LLM Client Abstraction

Create `llm/llm_client.py` supporting Ollama, OpenAI, and Anthropic backends:
```python
import httpx
from core.config.settings import settings

class LLMClient:
    def __init__(self, model: str = "presenter"):
        self.backend = settings.llm_backend
        model_map = {
            "presenter": settings.llm_model_presenter,
            "swarm": settings.llm_model_swarm,
            "world": settings.llm_model_world
        }
        self.model = model_map.get(model, settings.llm_model_presenter)

    async def complete(self, system: str, user: str, context: list = None) -> str:
        messages = [{"role": "system", "content": system}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user})

        if self.backend == "ollama":
            return await self._ollama(messages)
        elif self.backend == "openai":
            return await self._openai(messages)
        elif self.backend == "anthropic":
            return await self._anthropic(messages)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    async def _ollama(self, messages: list) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{settings.llm_base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False}
            )
            return resp.json()["message"]["content"]

    async def _openai(self, messages: list) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=self.model, messages=messages
        )
        return resp.choices[0].message.content

    async def _anthropic(self, messages: list) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic()
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        resp = await client.messages.create(
            model=self.model, max_tokens=2048, system=system, messages=user_msgs
        )
        return resp.content[0].text
```

---

## Phase 2 — Swarm Intelligence

**Goal:** Build the parallel branch-agent pool (Layer 2) that receives ranked hypotheses and pre-generates responses.  
**Duration:** 1–2 weeks  
**Deliverable:** N worker processes consuming branch hypotheses, solving them in parallel, and writing artifacts to cache.

### 2.1 — Branch Hypothesis Schema

Each branch is a structured prediction about the next world event. These are fed from Layer 3. Schema in `core/schemas/branches.py`:
```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class BranchStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SOLVED = "solved"
    PRUNED = "pruned"
    CONFIRMED = "confirmed"

class BranchHypothesis(BaseModel):
    branch_id: str
    session_id: str
    parent_branch_id: Optional[str] = None
    predicted_event_type: str
    predicted_content: str
    probability: float = Field(ge=0.0, le=1.0)
    depth: int = Field(ge=1, le=5)
    status: BranchStatus = BranchStatus.PENDING
    spawned_at: float  # Unix timestamp
    expires_at: float  # Unix timestamp (spawned_at + TTL)
    context_snapshot: list = []  # Message history at branch point
    metadata: dict = {}

class BranchArtifact(BaseModel):
    artifact_id: str
    branch_id: str
    session_id: str
    response_text: str
    reasoning_trace: Optional[str] = None
    readiness_score: float = Field(ge=0.0, le=1.0)
    token_cost: int = 0
    latency_ms: float = 0.0
    created_at: float
```

### 2.2 — Swarm Worker

Create `swarm/worker.py`:
```python
import asyncio
import hashlib
import json
import time
import uuid
from core.bus.streams import EventBus
from core.schemas.branches import BranchHypothesis, BranchArtifact, BranchStatus
from cache.readiness_cache import ReadinessCache
from llm.llm_client import LLMClient

class SwarmWorker:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.bus = EventBus()
        self.cache = ReadinessCache()
        self.llm = LLMClient(model="swarm")

    async def run(self):
        print(f"[Swarm Worker {self.worker_id}] Ready.")
        while True:
            messages = await self.bus.consume(
                stream="ppah:branches:ranked",
                group="swarm-group",
                consumer=self.worker_id,
                count=2
            )
            for stream, msgs in (messages or []):
                for msg_id, fields in msgs:
                    data = json.loads(fields[b"data"])
                    branch = BranchHypothesis(**data)
                    await self.solve_branch(branch, msg_id, stream)

    async def solve_branch(self, branch: BranchHypothesis, msg_id, stream):
        if branch.expires_at < time.time():
            await self.bus.ack(str(stream, "utf-8"), "swarm-group", str(msg_id, "utf-8"))
            return  # Branch expired before we could solve it

        start_ms = time.time() * 1000

        system_prompt = """You are a branch-agent in a precognitive reasoning system.
You are pre-solving a predicted future event BEFORE it happens.
Your job is to prepare the best possible response IF this event occurs.
Be thorough, accurate, and ready to deliver immediately.
Predicted event: {predicted_event}
Probability this event occurs: {probability:.0%}
Respond as if this event has just happened.""".format(
            predicted_event=branch.predicted_content,
            probability=branch.probability
        )

        response = await self.llm.complete(
            system=system_prompt,
            user=branch.predicted_content,
            context=branch.context_snapshot
        )

        latency = (time.time() * 1000) - start_ms

        artifact = BranchArtifact(
            artifact_id=str(uuid.uuid4()),
            branch_id=branch.branch_id,
            session_id=branch.session_id,
            response_text=response,
            readiness_score=self._score_readiness(branch, response),
            token_cost=len(response.split()),  # Rough estimate; replace with tiktoken
            latency_ms=latency,
            created_at=time.time()
        )

        await self.cache.store(branch, artifact)
        await self.bus.ack(str(stream, "utf-8"), "swarm-group", str(msg_id, "utf-8"))
        print(f"[Swarm {self.worker_id}] Solved branch {branch.branch_id[:8]} "
              f"(p={branch.probability:.2f}, latency={latency:.0f}ms)")

    def _score_readiness(self, branch: BranchHypothesis, response: str) -> float:
        """
        Readiness score formula:
        R = P(branch) × quality_factor × recency_factor
        quality_factor = min(len(response) / 200, 1.0)  # Response substance
        recency_factor = 1.0 (decays over time via cache TTL)
        """
        quality_factor = min(len(response.split()) / 150, 1.0)
        return branch.probability * quality_factor
```

### 2.3 — Swarm Pool Manager

Create `swarm/pool.py` to manage worker lifecycle:
```python
import asyncio
from swarm.worker import SwarmWorker

async def run_swarm_pool(n_workers: int = 5):
    workers = [SwarmWorker(f"worker-{i}") for i in range(n_workers)]
    await asyncio.gather(*[w.run() for w in workers])
```

---

## Phase 3 — World-Model Integration

**Goal:** Build the probability-tree engine (Layer 3) that watches resolved events and continuously generates ranked branch hypotheses for Layer 2.  
**Duration:** 2 weeks  
**Deliverable:** A live probability tree that updates per resolved event and publishes branch rankings.

### 3.1 — World State Schema

The world model maintains a session-level state representing the current known context and the outstanding branch tree. Schema in `core/schemas/world.py`:
```python
from pydantic import BaseModel
from typing import List, Dict, Optional
from enum import Enum

class EventType(str, Enum):
    USER_MESSAGE = "user_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_EVENT = "system_event"
    ENVIRONMENT_CHANGE = "environment_change"

class WorldState(BaseModel):
    session_id: str
    event_history: List[dict] = []  # Chronological resolved events
    context_window: List[dict] = []  # Recent events for LLM context
    active_branches: List[str] = []  # branch_ids currently in-flight
    confirmed_events: List[str] = []  # event_ids that resolved
    pruned_branches: List[str] = []
    last_updated: float = 0.0
    horizon_depth: int = 3  # Steps ahead to simulate

class ProbabilityNode(BaseModel):
    node_id: str
    parent_id: Optional[str] = None
    event_type: EventType
    predicted_content: str
    probability: float
    depth: int
    children: List[str] = []  # node_ids
    status: str = "pending"  # pending | active | confirmed | pruned
```

### 3.2 — World Model Service

Create `world_model/world_model.py`:
```python
import asyncio
import json
import time
import uuid
from core.bus.streams import EventBus
from core.schemas.world import WorldState, ProbabilityNode
from core.schemas.branches import BranchHypothesis
from core.config.settings import settings
from llm.llm_client import LLMClient

WORLD_STATE_TTL = 3600  # 1 hour session window

class WorldModel:
    def __init__(self):
        self.bus = EventBus()
        self.llm = LLMClient(model="world")
        self.sessions: dict[str, WorldState] = {}

    async def run(self):
        print("[WorldModel] Observing resolved events...")
        while True:
            messages = await self.bus.consume(
                stream="ppah:events:resolved",
                group="world-model-group",
                consumer="world-model-1",
                count=10
            )
            for stream, msgs in (messages or []):
                for msg_id, fields in msgs:
                    data = json.loads(fields[b"data"])
                    await self.process_resolved(data, msg_id, stream)

    async def process_resolved(self, resolved: dict, msg_id, stream):
        session_id = resolved["session_id"]
        state = self.sessions.get(session_id, WorldState(session_id=session_id))

        # Update world state with new confirmed event
        state.event_history.append(resolved)
        state.context_window = state.event_history[-20:]  # Keep last 20 events
        state.last_updated = time.time()
        self.sessions[session_id] = state

        # Generate new branch hypotheses
        branches = await self.generate_branches(state, resolved)

        # Publish ranked branches to swarm
        for branch in sorted(branches, key=lambda b: b.probability, reverse=True):
            await self.bus.publish("ppah:branches:ranked", branch.dict())

        await self.bus.ack(str(stream, "utf-8"), "world-model-group",
                           str(msg_id, "utf-8"))

    async def generate_branches(self, state: WorldState,
                                 resolved: dict) -> list[BranchHypothesis]:
        """
        Ask the LLM to simulate the top-K most probable next events.
        This is the core world-simulation step.
        """
        context_summary = self._summarize_context(state.context_window)
        k = settings.top_k_branches

        prompt = f"""You are a world-simulator. Given the conversation/event history below,
predict the {k} most probable next user messages or environment events.

For each prediction:
1. Describe what the user/environment is most likely to do next
2. Assign a probability between 0.0 and 1.0 (all must sum to ≤ 1.0)
3. Categorize as: user_message | tool_call | system_event | environment_change

Event history:
{context_summary}

Most recent resolved event:
{resolved.get('actual_content', '[unknown]')}

Respond ONLY as valid JSON array:
[
  {{"predicted_content": "...", "event_type": "user_message", "probability": 0.6}},
  {{"predicted_content": "...", "event_type": "user_message", "probability": 0.25}},
  ...
]"""

        raw = await self.llm.complete(
            system="You are a probabilistic world-simulator for an AI agent system. "
                   "Respond only with valid JSON.",
            user=prompt,
            context=[]
        )

        predictions = self._parse_predictions(raw)
        branches = []
        now = time.time()

        for pred in predictions[:k]:
            branch = BranchHypothesis(
                branch_id=str(uuid.uuid4()),
                session_id=state.session_id,
                predicted_event_type=pred.get("event_type", "user_message"),
                predicted_content=pred["predicted_content"],
                probability=float(pred["probability"]),
                depth=1,
                spawned_at=now,
                expires_at=now + settings.branch_ttl,
                context_snapshot=state.context_window[-10:]
            )
            branches.append(branch)

        return branches

    def _parse_predictions(self, raw: str) -> list:
        import re
        try:
            raw = raw.strip()
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(raw)
        except Exception as e:
            print(f"[WorldModel] Parse error: {e}")
            return []

    def _summarize_context(self, context_window: list) -> str:
        return "\n".join([
            f"[{e.get('source', '?')}] {e.get('actual_content', e.get('content', ''))[:200]}"
            for e in context_window[-10:]
        ])
```

---

## Phase 4 — Readiness & Cache Layer

**Goal:** Build the shared Readiness Cache that stores pre-solved branch artifacts and serves them to the Presenter.  
**Duration:** 3–5 days  
**Deliverable:** Redis-backed key-value store with semantic lookup, TTL management, and readiness scoring.

### 4.1 — Readiness Cache Service

Create `cache/readiness_cache.py`:
```python
import redis.asyncio as redis
import json
import time
import hashlib
from core.config.settings import settings
from core.schemas.branches import BranchHypothesis, BranchArtifact

class ReadinessCache:
    def __init__(self):
        self.client = redis.from_url(settings.redis_url)
        self.ttl = settings.branch_ttl

    def _branch_key(self, session_id: str, branch_id: str) -> str:
        return f"ppah:artifact:{session_id}:{branch_id}"

    def _content_hash(self, content: str) -> str:
        return hashlib.sha256(content.lower().strip().encode()).hexdigest()[:16]

    def _session_index_key(self, session_id: str) -> str:
        return f"ppah:index:{session_id}"

    async def store(self, branch: BranchHypothesis, artifact: BranchArtifact):
        """Store a pre-solved artifact and index it for lookup."""
        artifact_key = self._branch_key(branch.session_id, branch.branch_id)
        content_hash = self._content_hash(branch.predicted_content)

        payload = artifact.dict()
        payload["predicted_content"] = branch.predicted_content
        payload["content_hash"] = content_hash

        await self.client.setex(artifact_key, self.ttl, json.dumps(payload))

        # Index: content_hash → branch_id (for lookup by incoming event)
        index_key = self._session_index_key(branch.session_id)
        await self.client.hset(index_key, content_hash, branch.branch_id)
        await self.client.expire(index_key, self.ttl * 2)

    async def lookup(self, session_id: str, event_hash: str) -> BranchArtifact | None:
        """
        Lookup a pre-solved artifact for an incoming event.
        Returns the artifact if it exists and readiness score is sufficient.
        """
        index_key = self._session_index_key(session_id)
        branch_id = await self.client.hget(index_key, event_hash)

        if not branch_id:
            return None

        artifact_key = self._branch_key(session_id, str(branch_id, "utf-8"))
        raw = await self.client.get(artifact_key)

        if not raw:
            return None

        data = json.loads(raw)
        return BranchArtifact(**{k: v for k, v in data.items()
                                  if k in BranchArtifact.__fields__})

    async def get_all_for_session(self, session_id: str) -> list[BranchArtifact]:
        """Return all live artifacts for a session (for debugging/dashboard)."""
        index_key = self._session_index_key(session_id)
        all_entries = await self.client.hgetall(index_key)
        artifacts = []
        for _, branch_id in all_entries.items():
            artifact_key = self._branch_key(session_id, str(branch_id, "utf-8"))
            raw = await self.client.get(artifact_key)
            if raw:
                data = json.loads(raw)
                try:
                    artifacts.append(BranchArtifact(**{k: v for k, v in data.items()
                                                         if k in BranchArtifact.__fields__}))
                except Exception:
                    pass
        return artifacts

    async def invalidate_session(self, session_id: str):
        """Clear all artifacts for a session."""
        index_key = self._session_index_key(session_id)
        all_entries = await self.client.hgetall(index_key)
        keys_to_del = [index_key]
        for _, branch_id in all_entries.items():
            keys_to_del.append(self._branch_key(session_id, str(branch_id, "utf-8")))
        if keys_to_del:
            await self.client.delete(*keys_to_del)
```

---

## Phase 5 — Evaluation Harness

**Goal:** Build offline and live evaluation tooling to measure the system's predictive accuracy, readiness hit rate, and response quality.  
**Duration:** 1 week  
**Deliverable:** Automated evaluation suite with metrics, simulation replay, and dashboard output.

### 5.1 — Evaluation Metrics Specification

| Metric | Formula | Target | Notes |
|---|---|---|---|
| **Branch Hit Rate (BHR)** | confirmed_branches / total_branches_generated | ≥ 0.35 | 35% hit is excellent given entropy |
| **Cache Hit Rate (CHR)** | cache_hits / total_presenter_requests | ≥ 0.50 | Goal: Presenter pulls from cache ≥ 50% |
| **Readiness Score Mean (RSM)** | mean(artifact.readiness_score) | ≥ 0.60 | Quality-weighted probability |
| **P@1 Accuracy** | rank_1_confirmed / total_events | ≥ 0.25 | Was top prediction the actual event? |
| **Latency Saved (LS)** | mean(live_latency_ms) - mean(cache_latency_ms) | ≥ 300ms | Real user-perceived speedup |
| **Branch Depth Utility (BDU)** | value_delivered_at_depth_N / tokens_spent_at_depth_N | Maximize | Guides optimal depth setting |
| **Entropy Coverage** | unique_hypotheses / possible_next_states | Maximize | Penalizes overfitting to high-probability branches |
| **Stale Rate** | expired_artifacts / total_artifacts_created | ≤ 0.20 | High stale rate → increase TTL or reduce depth |
| **Token Efficiency (TE)** | value_delivered / total_tokens_consumed | Maximize | Tracks cost vs. utility |

### 5.2 — Evaluation Runner

Create `eval/evaluator.py`:
```python
import json
import time
from dataclasses import dataclass, field
from core.bus.streams import EventBus
from cache.readiness_cache import ReadinessCache

@dataclass
class EvalSession:
    session_id: str
    total_events: int = 0
    cache_hits: int = 0
    branch_hits: int = 0
    total_branches: int = 0
    live_latencies: list = field(default_factory=list)
    cache_latencies: list = field(default_factory=list)
    readiness_scores: list = field(default_factory=list)
    rank1_hits: int = 0

    def branch_hit_rate(self) -> float:
        return self.branch_hits / max(self.total_branches, 1)

    def cache_hit_rate(self) -> float:
        return self.cache_hits / max(self.total_events, 1)

    def readiness_score_mean(self) -> float:
        return sum(self.readiness_scores) / max(len(self.readiness_scores), 1)

    def latency_saved_ms(self) -> float:
        if not self.live_latencies or not self.cache_latencies:
            return 0.0
        return (sum(self.live_latencies) / len(self.live_latencies)) - \
               (sum(self.cache_latencies) / len(self.cache_latencies))

    def p_at_1(self) -> float:
        return self.rank1_hits / max(self.total_events, 1)

    def report(self) -> dict:
        return {
            "session_id": self.session_id,
            "branch_hit_rate": round(self.branch_hit_rate(), 3),
            "cache_hit_rate": round(self.cache_hit_rate(), 3),
            "readiness_score_mean": round(self.readiness_score_mean(), 3),
            "latency_saved_ms": round(self.latency_saved_ms(), 1),
            "p_at_1": round(self.p_at_1(), 3),
            "total_events": self.total_events,
            "total_branches": self.total_branches,
        }
```

---

## Phase 6 — MiroFish Integration

**Goal:** Replace the WorldModel's in-house branch generator with MiroFish's parallel agent simulation engine for richer and more scalable probability tree generation.  
**Duration:** 1–2 weeks  
**Prerequisites:** MiroFish service running at `MIROFISH_URL` (see https://github.com/666ghj/MiroFish)

### 6.1 — MiroFish Adapter

MiroFish's native role is simulating what other agents do. The reframe needed: instead of simulating **agent actions**, configure MiroFish's universe agents to simulate **external world/user behaviors**. Each universe agent is assigned a behavioral persona (e.g., "a curious user who wants to go deeper," "a user who wants to change topics," "a system that has just returned a tool result"). These personas produce the branch hypotheses.

Create `mirofish/adapter.py`:
```python
import httpx
import json
from core.config.settings import settings
from core.schemas.world import WorldState
from core.schemas.branches import BranchHypothesis
import uuid, time

class MiroFishAdapter:
    def __init__(self):
        self.base_url = settings.mirofish_url

    async def generate_branches(self, state: WorldState,
                                 resolved_event: dict,
                                 n_branches: int = 5) -> list[BranchHypothesis]:
        """
        Submit world state to MiroFish universe for parallel simulation.
        MiroFish runs N agents, each playing the role of the next world-event.
        """
        payload = {
            "universe_id": state.session_id,
            "context": state.context_window[-10:],
            "last_event": resolved_event,
            "agent_personas": self._generate_personas(n_branches),
            "simulation_depth": 1,
            "output_format": "branch_hypotheses"
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{self.base_url}/api/simulate",
                json=payload
            )
            resp.raise_for_status()
            results = resp.json().get("simulations", [])

        now = time.time()
        branches = []
        for r in results:
            branch = BranchHypothesis(
                branch_id=str(uuid.uuid4()),
                session_id=state.session_id,
                predicted_event_type=r.get("event_type", "user_message"),
                predicted_content=r["predicted_content"],
                probability=float(r.get("probability", 0.2)),
                depth=1,
                spawned_at=now,
                expires_at=now + settings.branch_ttl,
                context_snapshot=state.context_window[-10:],
                metadata={"source": "mirofish", "persona": r.get("persona")}
            )
            branches.append(branch)

        return sorted(branches, key=lambda b: b.probability, reverse=True)

    def _generate_personas(self, n: int) -> list[dict]:
        """
        World-simulator personas — not agent personas.
        Each represents a type of next external event.
        """
        base_personas = [
            {"id": "deep_dive", "role": "user",
             "behavior": "wants to explore the current topic in more depth"},
            {"id": "topic_shift", "role": "user",
             "behavior": "wants to pivot to a related but different topic"},
            {"id": "clarification", "role": "user",
             "behavior": "needs clarification on something just said"},
            {"id": "action_request", "role": "user",
             "behavior": "wants to take an action based on the response"},
            {"id": "tool_return", "role": "system",
             "behavior": "a background tool or API has returned a result"},
            {"id": "pushback", "role": "user",
             "behavior": "disagrees or wants to challenge the response"},
        ]
        return base_personas[:n]
```

### 6.2 — Swap World Model Backend

Update `world_model/world_model.py` to use MiroFish when available:
```python
# At top of WorldModel class __init__:
from mirofish.adapter import MiroFishAdapter

self.mirofish = MiroFishAdapter()
self.use_mirofish = True  # Toggle via env var in production

# In generate_branches():
if self.use_mirofish:
    try:
        return await self.mirofish.generate_branches(state, resolved, settings.top_k_branches)
    except Exception as e:
        print(f"[WorldModel] MiroFish unavailable ({e}), falling back to LLM generator")

# Fall through to in-house LLM-based generator as fallback
```

---

## API Specification

The HTTP API Gateway exposes PPAH to external clients. All endpoints use JSON. Authentication via Bearer token (implement as needed for your deployment).

### Base URL
```
http://localhost:8080/api/v1
```

### Endpoints

#### POST /sessions
Create a new session context.
```json
// Request
{
  "user_id": "user-abc123",
  "metadata": { "client": "web", "timezone": "America/Los_Angeles" }
}

// Response 201
{
  "session_id": "sess-uuid-here",
  "created_at": 1745000000.0,
  "status": "active"
}
```

#### POST /sessions/{session_id}/events
Submit an inbound event to the system (main entry point for user input).
```json
// Request
{
  "content": "Can you explain how backpropagation works?",
  "event_type": "user_message",
  "context": []
}

// Response 200
{
  "event_id": "evt-uuid",
  "session_id": "sess-uuid",
  "received_at": 1745000001.2,
  "source": "cache_hit",
  "response": "Backpropagation is the algorithm used to compute gradients..."
}
```

#### GET /sessions/{session_id}/branches
Retrieve current live branch hypotheses for a session (debug/dashboard use).
```json
// Response 200
{
  "session_id": "sess-uuid",
  "branches": [
    {
      "branch_id": "br-uuid-1",
      "predicted_content": "Follow-up asking about gradient descent",
      "probability": 0.58,
      "status": "solved",
      "readiness_score": 0.71
    },
    {
      "branch_id": "br-uuid-2",
      "predicted_content": "Question about vanishing gradient problem",
      "probability": 0.27,
      "status": "active",
      "readiness_score": null
    }
  ]
}
```

#### GET /sessions/{session_id}/metrics
Return evaluation metrics for a session.
```json
// Response 200
{
  "session_id": "sess-uuid",
  "branch_hit_rate": 0.41,
  "cache_hit_rate": 0.62,
  "readiness_score_mean": 0.67,
  "latency_saved_ms": 420.5,
  "p_at_1": 0.31,
  "total_events": 18,
  "total_branches": 54
}
```

#### DELETE /sessions/{session_id}
Terminate a session and clear its cache.
```json
// Response 200
{ "session_id": "sess-uuid", "status": "terminated" }
```

#### POST /admin/config
Update runtime parameters without restart.
```json
// Request
{
  "top_k_branches": 7,
  "branch_ttl": 45,
  "readiness_threshold": 0.60
}

// Response 200
{ "status": "updated", "effective_at": 1745000010.0 }
```

---

## Event Schemas

All events flowing through the system conform to these schemas.

### InboundEvent
```json
{
  "event_id": "string (uuid)",
  "session_id": "string",
  "content": "string — raw user/environment input",
  "event_type": "user_message | tool_result | system_event | environment_change",
  "content_hash": "string (sha256[:16] of normalized content)",
  "context": "array of {role, content} message history",
  "received_at": "float (unix timestamp)",
  "metadata": "object (optional)"
}
```

### ResolvedEvent
```json
{
  "event_id": "string (uuid)",
  "session_id": "string",
  "original_event": "InboundEvent object",
  "actual_content": "string — confirmed content",
  "response_delivered": "string — what the Presenter sent",
  "source": "cache_hit | live_inference",
  "resolved_at": "float (unix timestamp)",
  "latency_ms": "float"
}
```

### BranchHypothesis
```json
{
  "branch_id": "string (uuid)",
  "session_id": "string",
  "parent_branch_id": "string | null",
  "predicted_event_type": "string",
  "predicted_content": "string — predicted next event description",
  "probability": "float [0.0–1.0]",
  "depth": "integer [1–5]",
  "status": "pending | active | solved | pruned | confirmed",
  "spawned_at": "float",
  "expires_at": "float",
  "context_snapshot": "array (last N resolved events)",
  "metadata": "object"
}
```

### BranchArtifact
```json
{
  "artifact_id": "string (uuid)",
  "branch_id": "string",
  "session_id": "string",
  "response_text": "string — pre-solved response",
  "reasoning_trace": "string | null",
  "readiness_score": "float [0.0–1.0]",
  "token_cost": "integer",
  "latency_ms": "float",
  "created_at": "float"
}
```

### WorldState
```json
{
  "session_id": "string",
  "event_history": "array of ResolvedEvent",
  "context_window": "array (last 20 events)",
  "active_branches": "array of branch_ids",
  "confirmed_events": "array of event_ids",
  "pruned_branches": "array of branch_ids",
  "last_updated": "float",
  "horizon_depth": "integer [1–5]"
}
```

---

## Module Contracts

### Module Responsibility Matrix

| Module | Layer | Reads From | Writes To | Owns |
|---|---|---|---|---|
| `presenter` | 1 | `ppah:events:inbound` | `ppah:events:resolved` | Response delivery |
| `readiness_cache` | 1/2 | Redis keys | Redis keys | Artifact storage |
| `swarm_worker` | 2 | `ppah:branches:ranked` | Redis artifacts | Branch solving |
| `world_model` | 3 | `ppah:events:resolved` | `ppah:branches:ranked` | Probability tree |
| `mirofish_adapter` | 3 | WorldState | BranchHypothesis list | World simulation |
| `eval` | X-layer | All streams (tap) | Metrics store | Measurement |
| `api_gateway` | External | HTTP | `ppah:events:inbound` | External interface |

### Invariants (Never Violate)

1. **Presenter never waits for swarm.** The Presenter's response time is bounded by either cache lookup or single-model inference — never by swarm availability.
2. **Branches never influence confirmed reality.** The cache can only be *read* by the Presenter. The Presenter decides what to deliver. Swarm artifacts are suggestions, not commitments.
3. **World state flows one direction per cycle.** ResolvedEvent → WorldModel → BranchHypothesis → Swarm → Artifact → Cache → Presenter. No short-circuits.
4. **Expired artifacts are never served.** The Presenter must check TTL before serving a cached artifact, even if readiness_score is high.
5. **Each session has exactly one WorldModel state.** Multi-session systems shard by session_id; cross-session leakage of branch hypotheses is a correctness bug.

---

## Speculative Outcomes & Refinement Roadmap

The following outlines predicted behaviors, failure modes, and revision cycles based on the FLARE architecture analysis and general knowledge of LLM agent systems.

### Expected Initial Outcomes (First 4–6 Weeks)

**What will likely work well:**
- Cache hit rate will be higher than expected for structured task domains (coding assistants, customer support, tutoring) where user behavior is predictable. Expect CHR of 55–70% in these domains.
- The Presenter will feel measurably more responsive immediately, because cache hits are near-zero latency.
- The swarm will produce useful artifacts even when exact-match fails, because the LLM responses are semantically coherent around the predicted topic.

**What will likely underperform initially:**
- P@1 accuracy in open-ended or creative conversations will be low (15–25%). This is expected and acceptable — the system is a hedge, not a prediction oracle.
- Branch depth > 2 will have very low confirmation rates. Depth 3+ artifacts will mostly go stale. Start with `BRANCH_DEPTH=2` in production.
- MiroFish personas will need significant prompt tuning before they produce high-quality behavioral simulations. Plan 2–3 revision cycles.

### Revision Cycle 1 — Semantic Matching (Weeks 4–6)

The first major limitation will be that cache lookup relies on exact content hash matching. A user asking "how does backprop work?" will miss an artifact pre-solved for "explain backpropagation." 

**Fix:** Replace hash-based lookup with embedding similarity search.
- Embed each predicted_content at artifact creation time.
- Store vector in Redis using `redis-stack` with vector similarity (HNSW index).
- At lookup time, embed the inbound event content, find top-1 artifact with cosine similarity > 0.85.
- Libraries: `sentence-transformers` (local) or OpenAI embeddings.

### Revision Cycle 2 — Branch Quality Feedback Loop (Weeks 6–8)

After collecting 500+ sessions of data, build a feedback loop that trains the world model to generate better branch hypotheses by learning which past hypotheses were confirmed.

**Implementation:**
- Log every branch + outcome (confirmed/pruned) to a PostgreSQL table.
- Weekly: Fine-tune branch generation prompts using confirmed examples as few-shot context.
- Monthly: Fine-tune the swarm model on confirmed branch + high-quality response pairs.

### Revision Cycle 3 — Adaptive Depth Control (Weeks 8–10)

`BRANCH_DEPTH` should not be a static config value. Different session types warrant different depths.

**Fix:** Implement a depth heuristic:
```
depth = 1 + floor(session_complexity_score / 0.3)
session_complexity_score = running average of confirmed P@1 * predictability
```
For highly predictable sessions (repetitive queries, FAQ-like), allow depth 3.
For open-ended creative sessions, cap at depth 1.

### Revision Cycle 4 — Multi-Turn Branch Linking (Weeks 10–12)

Currently, each branch is a flat prediction of the next single event. In multi-turn conversations, the system can do much better by chaining branch predictions: if Branch A is confirmed, what is the most likely Branch B given Branch A happened?

**Fix:** Implement BranchTree with parent_branch_id linking:
- When Branch A is confirmed, immediately spawn A-children branches based on that confirmation.
- These "conditional branches" have lower initial spawn cost because their context is tighter.
- This enables the system to begin simulating 2–3 turns ahead with high accuracy in structured domains.

### Revision Cycle 5 — Safety Layer (Weeks 12–16)

The speculative nature of the swarm creates risk: a pre-solved artifact might be based on a hypothesized event that is factually wrong or contextually harmful. Before an artifact is served, it must pass a safety check.

**Implementation:**
- Add a `SafetyEvaluator` module that runs lightweight checks on artifacts before cache storage:
  - Hallucination check: Does the artifact contradict confirmed world state?
  - Toxicity check: Standard content filter pass.
  - Factual drift check: Does the artifact introduce new claimed facts not in context?
- Artifacts that fail safety checks are stored with `readiness_score = 0` (will never be served).

### Long-Term Research Directions

- **Online learning of world-model priors:** Rather than prompting a general LLM to simulate the world, fine-tune a small specialized model (1–3B params) on session histories to serve as the world-model. This dramatically reduces token cost and increases prediction accuracy.
- **Multi-agent adversarial branching:** Introduce one "contrarian" swarm agent per cycle whose job is to generate the *least* predicted next event. This dramatically improves entropy coverage and catches edge cases.
- **Cross-session branch transfer:** In domains with many similar users (e.g., a customer support bot), branch hypotheses confirmed in Session A can pre-warm the cache for Session B if their context is sufficiently similar. This is a form of population-level predictive caching.
- **Streaming branch updates:** Deliver partially-solved artifacts to the Presenter as they are generated (token streaming), rather than waiting for full artifact completion. This allows the Presenter to begin delivering a response within 50–100ms even for cache misses.

---

*Implementation plan authored in companion with the Precognitive Agent Harness System Spec and Architecture Overview. All code samples are reference implementations — adapt to your runtime, model selection, and infrastructure constraints.*

