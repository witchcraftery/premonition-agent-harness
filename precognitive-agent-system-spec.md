# Precognitive Agent Harness Formal System Specification

## Overview

This specification defines a receding-horizon agent architecture that separates present-time delivery from near-future preparation and deeper possibility-space simulation. The design is grounded in the planning-centric view that step-wise reasoning alone is insufficient for long-horizon decision quality, because locally good actions can be globally poor when future consequences are not explicitly evaluated [cite:1]. The attached single-page system concept already captures the key separation into Presenter, Swarm Readiness Engine, and Probability World-Model Engine, with observed truth always outranking simulated possibilities [cite:4].

## System goals

- Predict likely next events rather than only likely next tokens.
- Prepare useful artifacts before the next event arrives.
- Keep speculative branches structurally sparse until evidence justifies expansion.
- Ensure user-facing output remains grounded in observed reality rather than simulated certainty.
- Support selective escalation into richer world simulation for high-uncertainty or high-impact branches.

## Design principles

- **Observed truth wins.** Branch outputs cannot become user-facing facts until reality confirms them [cite:4].
- **Receding-horizon commitment.** The system commits only to the next action while continuously replanning from fresh observations, which mirrors the planning discipline highlighted in the FLARE paper [cite:1].
- **Sparse branch representation.** Most futures should be encoded as compact event signatures, not long narrative continuations.
- **Selective compute.** Cheap branch prediction runs frequently; expensive simulation is triggered only when expected readiness value justifies it [cite:4].
- **Preparedness over prophecy.** The system is evaluated by readiness and recovery, not by dramatic claims of perfect foresight [cite:4].

## High-level architecture

| Layer | Purpose | Inputs | Outputs |
|---|---|---|---|
| Presenter / Face Agent | Respond from current observed reality | Current state, branch cache, tools | User-facing response, action execution |
| Event Abstraction Layer | Compress raw reality into causal signatures | Conversation turns, sensor streams, tool outputs, environment signals | Event signatures, salience tags, uncertainty markers |
| Swarm Readiness Engine | Generate top-k next-event branches and low-cost preparations | Event signatures, world state, priors | Prepared drafts, plans, checks, fallbacks |
| Probability World-Model Engine | Maintain and update the live possibility tree | Event history, branch outcomes, priors, optional simulation | Ranked branches, lineage graph, calibration stats |
| Selective Simulation Engine | Run richer rollouts for hard branches | Escalated branch specs, actor models, domain constraints | Refined branch trajectories, impact estimates |
| Safety and Evaluation Wrapper | Gate actions and score system behavior | Prepared outputs, actual outcomes, policy rules | Safety decisions, metrics, audit logs |

## Module specification

### 1. Presenter service

**Responsibility**
- Serve as the visible system interface.
- Query prepared branch artifacts before attempting live generation.
- Fall back to live reasoning when branch confidence is low, stale, or missing [cite:4].

**Core functions**
- `resolve_response(event_signature)`
- `stream_response(user_context)`
- `emit_ground_truth(observed_event)`
- `request_live_reasoning(context)`

**Input contract**
- Current observation bundle.
- Top prepared artifacts from readiness cache.
- Confidence and freshness metadata.

**Output contract**
- User-visible response.
- Ground-truth event for downstream pruning.
- Audit record including cache hit or miss.

### 2. Event abstraction service

**Responsibility**
- Turn raw observations into compact branchable state.
- Prevent the branch engine from overfitting to surface detail.

**Core functions**
- `encode_event(raw_observation)`
- `extract_entities(raw_observation)`
- `detect_salience(raw_observation)`
- `estimate_uncertainty(raw_observation)`

**Output fields**
- Actors.
- Roles.
- Intent cues.
- Hazard flags.
- public/private context.
- temporal urgency.
- confidence score.

### 3. Swarm readiness engine

**Responsibility**
- Generate and rank top-k near-future branches.
- Attach lightweight preparations to each branch.
- Expand only the highest-value candidates [cite:4].

**Core functions**
- `generate_branches(event_signature, k)`
- `prepare_artifacts(branch)`
- `score_readiness(branch)`
- `publish_cache_entries(branches)`

**Prepared artifacts**
- Draft responses.
- Tool plans.
- Recovery paths.
- Safety checks.
- Citation stubs.
- Alternate fallback messaging.

### 4. Probability world-model engine

**Responsibility**
- Maintain the live tree of likely next realities.
- Update priors from actual outcomes.
- Preserve branch lineage while pruning dead paths [cite:4].

**Core functions**
- `update_tree(observed_event)`
- `prune_dead_branches(observed_event)`
- `advance_horizon()`
- `rank_branches()`
- `record_lineage()`
- `update_calibration(predicted, actual)`

**Planning alignment**
- The engine should support explicit lookahead, branch scoring, and backward value updates, because the FLARE paper shows these are key to overcoming step-wise myopia in long-horizon settings [cite:1].

### 5. Selective simulation engine

**Responsibility**
- Run richer multi-actor or high-impact rollouts only when justified.
- Use MiroFish-style parallel simulation as a world-model coprocessor rather than the default path for every turn [cite:2][cite:4].

**Trigger conditions**
- High uncertainty.
- High downside if unprepared.
- Multi-actor social dynamics.
- Second-order effects dominate.
- Safety-critical domains.

**Core functions**
- `simulate_branch(branch_spec, depth)`
- `evaluate_trajectory(trajectory)`
- `return_refined_readiness(branch_id)`

### 6. Safety and evaluation wrapper

**Responsibility**
- Prevent speculative artifacts from becoming unsafe actions.
- Measure whether the system is actually more prepared when reality arrives [cite:4].

**Core functions**
- `gate_action(proposed_action)`
- `score_safety(branch, action, outcome)`
- `compute_metrics(run_id)`
- `emit_audit_log(event)`

## API specification

### 1. Observation ingestion API

```json
POST /v1/observe
{
  "session_id": "sess_001",
  "timestamp": "2026-04-15T00:18:00-07:00",
  "source": "conversation|video|sensor|tool",
  "raw_observation": {
    "text": "user asks about branch pruning",
    "tool_events": [],
    "env_signals": []
  },
  "metadata": {
    "importance": 0.74,
    "latency_budget_ms": 400,
    "domain": "assistant"
  }
}
```

Response:

```json
{
  "event_id": "evt_001",
  "event_signature_id": "sig_001",
  "status": "accepted"
}
```

### 2. Event abstraction API

```json
POST /v1/abstract
{
  "event_id": "evt_001"
}
```

Response:

```json
{
  "event_signature_id": "sig_001",
  "signature": {
    "actors": ["user", "assistant"],
    "intent": "architecture-expansion",
    "salience": ["formal-spec", "html-update"],
    "hazard_flags": [],
    "urgency": "medium",
    "confidence": 0.83
  }
}
```

### 3. Branch generation API

```json
POST /v1/branches/generate
{
  "event_signature_id": "sig_001",
  "top_k": 5,
  "horizon": 3,
  "mode": "lightweight"
}
```

Response:

```json
{
  "branch_set_id": "brset_001",
  "branches": [
    {
      "branch_id": "br_001",
      "event_hypothesis": "user asks for API schema details",
      "probability": 0.34,
      "impact": 0.68,
      "readiness_value": 0.71,
      "state": "prepared"
    }
  ]
}
```

### 4. Prepared artifact lookup API

```json
GET /v1/cache/resolve?event_signature_id=sig_001
```

Response:

```json
{
  "cache_hit": true,
  "confidence": 0.81,
  "freshness_ms": 920,
  "artifact": {
    "artifact_id": "art_001",
    "type": "response_draft",
    "content_ref": "cache://artifacts/art_001"
  }
}
```

### 5. World-model update API

```json
POST /v1/world/update
{
  "event_signature_id": "sig_001",
  "observed_outcome": {
    "actual_event": "user requested formal spec and html update",
    "matched_branch_id": "br_004"
  }
}
```

Response:

```json
{
  "tree_revision": 98,
  "pruned_branch_count": 4,
  "surviving_branch_count": 2,
  "calibration_delta": -0.03
}
```

### 6. Simulation escalation API

```json
POST /v1/simulate/escalate
{
  "branch_id": "br_009",
  "reason": "high-impact multi-actor uncertainty",
  "depth": 4,
  "engine": "mirofish"
}
```

Response:

```json
{
  "simulation_id": "sim_001",
  "status": "running",
  "expected_budget_ms": 2400
}
```

### 7. Safety gate API

```json
POST /v1/safety/gate
{
  "proposed_action": {
    "action_type": "external_actuation",
    "source_branch_id": "br_009",
    "confidence": 0.62
  }
}
```

Response:

```json
{
  "decision": "defer_to_live_reasoning",
  "reason_codes": ["low_confidence", "high_impact_domain"],
  "requires_human_review": false
}
```

## Event schema

### Canonical event schema

```json
{
  "$schema": "https://precog.system/schema/event/v1",
  "event_id": "evt_001",
  "session_id": "sess_001",
  "timestamp": "2026-04-15T00:18:00-07:00",
  "source": "conversation",
  "domain": "assistant",
  "actors": [
    {
      "actor_id": "user_1",
      "role": "user",
      "state": ["inquiring", "iterative"]
    }
  ],
  "observation": {
    "text": "Could you put together a formal system spec?",
    "attachments": [],
    "env_signals": []
  },
  "derived": {
    "intent": "system_design_request",
    "salience": ["api", "schema", "visualization"],
    "hazard_flags": [],
    "urgency": "medium",
    "confidence": 0.89
  }
}
```

### Branch node schema

```json
{
  "$schema": "https://precog.system/schema/branch-node/v1",
  "branch_id": "br_001",
  "parent_branch_id": "root",
  "lineage_path": ["root", "br_001"],
  "horizon_step": 1,
  "event_signature": "user_requests_more_api_detail",
  "probability": 0.34,
  "confidence_interval": [0.28, 0.39],
  "impact": 0.68,
  "readiness_value": 0.71,
  "expansion_state": "dormant",
  "trigger_conditions": ["api_section_visible", "user asks follow-up"],
  "artifact_refs": ["art_001", "art_004"],
  "safety_tags": ["low_risk"],
  "observed_match": false,
  "created_at": "2026-04-15T00:18:01-07:00"
}
```

### Prepared artifact schema

```json
{
  "$schema": "https://precog.system/schema/artifact/v1",
  "artifact_id": "art_001",
  "branch_id": "br_001",
  "artifact_type": "response_draft",
  "freshness_ms": 920,
  "grounding_state": "speculative",
  "utility_estimate": 0.77,
  "cost_estimate": 0.12,
  "payload": {
    "summary": "API details with example payloads",
    "tool_plan": ["read existing html", "write updated html", "share file"]
  }
}
```

### World-state revision schema

```json
{
  "$schema": "https://precog.system/schema/world-state/v1",
  "tree_revision": 98,
  "session_id": "sess_001",
  "root_event_signature_id": "sig_001",
  "active_branches": ["br_001", "br_004"],
  "dormant_branches": ["br_002", "br_003", "br_005"],
  "pruned_branches": ["br_006"],
  "last_observed_event_id": "evt_004",
  "updated_at": "2026-04-15T00:18:02-07:00"
}
```

## Internal dataflow

1. Observation arrives from conversation, sensor feed, video frame event, or tool result.
2. Event abstraction compresses the moment into a minimal causal signature.
3. Swarm readiness engine generates top-k branches and attaches low-cost preparations.
4. Probability world-model stores the branch set and tracks lineage.
5. Presenter checks for a strong prepared artifact before doing live reasoning.
6. Actual event arrives and is compared to predicted branches.
7. Dead branches are pruned, surviving branches are updated, and calibration is recomputed.
8. High-impact uncertainty triggers selective simulation for deeper rollouts [cite:2][cite:4].

## Scoring functions

### Readiness value

A simple first-pass scoring function can be:

\[
R(b) = P(b) 	imes I(b) 	imes U(b) - C(b)
\]

Where:
- \(P(b)\) is branch probability.
- \(I(b)\) is branch impact if it happens.
- \(U(b)\) is usefulness of pre-preparation.
- \(C(b)\) is cost of false preparation.

This formalizes the V3 design thesis that not every plausible branch is worth preparing for, and that the right objective is expected readiness rather than narrative completeness [cite:4].

### Escalation score

\[
E(b) = P(b) 	imes I(b) 	imes H(b) 	imes Q(b)
\]

Where:
- \(H(b)\) is uncertainty or entropy.
- \(Q(b)\) is second-order interaction complexity.

Branches with high escalation scores are candidates for MiroFish-style deeper simulation [cite:2].

## Evaluation metrics

### Prediction quality

| Metric | Definition | Why it matters |
|---|---|---|
| Top-1 next-event accuracy | Fraction of turns where the highest-ranked branch matches reality | Measures direct predictive usefulness [cite:4] |
| Top-3 branch recall | Fraction of turns where actual outcome appears in top 3 branches | Tests whether sparse branching still covers reality [cite:4] |
| Calibration error | Gap between predicted probabilities and observed frequencies | Prevents overconfident branch systems |
| Branch diversity | Distinctness among active branches | Reduces collapse into near-duplicate futures |

### Readiness quality

| Metric | Definition | Why it matters |
|---|---|---|
| Time-to-useful-response | Time from event arrival to a materially helpful response | Core user-visible readiness measure [cite:4] |
| Cache hit usefulness | Share of cache hits judged helpful enough to use | Distinguishes useful preparation from noisy preparation |
| False-prep cost | Resource cost spent preparing branches that never occur | Keeps the system efficient |
| Recovery speed | Time to recover when the prepared branch was wrong | Measures graceful degradation |

### Safety quality

| Metric | Definition | Why it matters |
|---|---|---|
| Unsafe speculative exposure rate | Rate at which speculative content is surfaced as fact | Tests the observed-truth rule [cite:4] |
| Safety override precision | Fraction of safety interventions that were actually warranted | Prevents overly conservative blocking |
| Safety override recall | Fraction of hazardous cases correctly intercepted | Protects high-impact domains |
| Human-escalation appropriateness | Quality of defer-to-human decisions | Important in actuation or medical-like settings |

### System efficiency

| Metric | Definition | Why it matters |
|---|---|---|
| Average tokens per branch | Mean token cost of branch generation | Tests sparse-branch discipline |
| Mean simulation escalation rate | How often deep simulation is invoked | Guards against overuse of heavy simulation |
| Latency budget compliance | Share of turns staying within service budget | Essential in real-time settings |
| Prepared artifact reuse rate | Rate at which prepared artifacts are actually consumed | Indicates whether readiness work pays off [cite:4] |

## Evaluation protocols

### Offline replay

Use logged conversation or environment traces to replay the system against known outcomes. This is especially useful because the FLARE paper evaluates planning under controlled deterministic settings, which makes outcome comparison clear and reproducible [cite:1].

### Video-grounded foresight testing

Use recorded video scenarios to feed sequential reality into the observation layer, then score branch prediction against what actually happens next. This is a useful low-risk testbed for social-world branch prediction because shared observable reality unfolds in time and can be annotated after the fact.

### Safety-critical simulation

For domains like driving or robotics, use constrained simulation environments to test whether sparse branch prediction and escalation improve response quality without violating latency or safety guardrails.

## Reference implementation roadmap

### Phase 1
- Presenter service.
- Event abstraction layer.
- Branch cache.
- Metrics logging.

### Phase 2
- Lightweight top-k branch generator.
- Prepared artifact generation.
- Calibration dashboard.

### Phase 3
- World-model lineage store.
- Backward value updates.
- Branch pruning and horizon advancement.

### Phase 4
- Selective MiroFish integration for escalated branches.
- Safety gate policies.
- Offline replay and benchmark suite.

## Implementation notes for the updated single-page HTML

The site should visualize the system structurally rather than theatrically. The spark tree animation should show a root event, shallow branch sparks, branch dormancy, and one highlighted lineage that survives contact with reality. The diagrams should emphasize that the external world feeds observations into the system and that only observed events activate deeper branch progression [cite:4].

## Conclusion

The strongest version of this architecture is not a prophecy machine. It is a sparse, calibrated readiness engine that keeps a live tree of near-futures, prepares selectively, and updates itself only when reality resolves uncertainty. That framing is consistent with the planning lessons from FLARE and with the disciplined modularity already present in the V3 concept [cite:1][cite:4].
