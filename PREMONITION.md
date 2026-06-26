# Premonition Backend Playbook

The frontend LLM performs the conversation. The Premonition Backend performs the
rehearsal.

## Operating Loop

1. Observe the current context: conversation, policy, tool state, and known user goal.
2. Generate the top likely next-event branches.
3. Prepare one artifact per useful branch.
4. Filter unsafe, stale, or overconfident artifacts.
5. When the next event arrives, select the best prepared artifact.
6. Log the outcome and grade the premonition.
7. Use miss analysis to improve the next branch generator, scorer, or artifact template.

## Safety Rules

- Speculation stays inside the backend until observed truth selects a branch.
- Prepared artifacts must carry policy checks and freshness status.
- The frontend should receive compact packets, not the entire probability tree by default.
- Unsafe branches should lower confidence and create benchmark evidence, not user-facing claims.

## Premonition Packet

The backend hands the frontend a packet with:

- observed context
- matched branch and confidence
- prepared artifact
- policy checks
- freshness status
- unsafe flag

The packet is readiness context. It is not a command to answer blindly.
