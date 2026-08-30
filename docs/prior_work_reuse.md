# Prior-work reuse map

## Supplied paper

The harness directly reuses the supplied paper's coding-agent scenario structure where that structure does not depend on coding: six agents, 30 action rounds, five initially clean workers, a direct-message-only origin, separate initial tasks, private memory, a shared task queue, a fully connected graph, a bridge graph, a final context-wipe warning, target-specific 0–3 judging, and a strict score-3 threshold.

It also reuses the paper's robustness axes: benign versus misaligned content, host-model variation, task pressure, network topology, memory persistence, explicit warning defense, and a fixed descriptive taxonomy for resonance, networks, consciousness or continuity, and roleplay themes. Paper-aligned real-model manifests poll inboxes after four action rounds. The warning text is reproduced from the paper's defensive-prompt appendix.

The source-removal experiment borrows from the paper's virus-chain logic. The origin is disabled after round 5, clean-agent contexts are wiped at rounds 10 and 20, and only persistent environmental state remains. This tests whether contagion has become decentralized rather than measuring repeated source persuasion.

## Choices not reused

The paper's evolved seed is intentionally not reused because the present question is spontaneous online strategy creation. The origin gets a target and population objective only. Its realized messages are the object of analysis.

The coding shell and unrestricted file operations are not reused because the target claim is non-coding generalization. Immutable synthetic evidence, shared report documents, task submission, direct messages, and private memory provide analogous collaboration pressure without code execution.

The automatic copying mechanisms studied in retrieval-memory prompt injection work, cited in the supplied paper as `gu_agent_2024`, `cohen_here_2025`, and `lee_prompt_2024`, are not used. Messages are not inserted into private memory automatically. A clean agent must choose to write memory and choose to send a new message, which preserves the distinction between architectural replication and agent-mediated adoption.

The forced adversarial-string repetition paradigm cited as `yu2025infecting` is also not used. It would test copying while disabling normal agentic behavior. Here, assigned document work continues and task completion is measured as a guardrail.

## Additional principles carried forward

The distinction between overt persuasion and subliminal transfer, discussed through `weckbecker2026thoughtvirusviralmisalignment` and `cloud_subliminal_2025`, motivates retaining complete visible messages and making no claim about hidden or subliminal mechanisms.

Topology-sensitive error and influence propagation, connected in the supplied paper to `shen_understanding_2025`, motivates graph-distance attribution and excluding fully connected graphs from a distance-two endpoint.

The multi-agent prompt-injection and facility-simulation work cited as `peigne-lefebvre_multi-agent_2025` motivates testing ordinary collaboration channels, while the present tool set is richer and retains independent task work.

The defense results and red-teaming discussion cited as `potts_red-teaming_2026` motivate treating warning prompts as an intervention, not merely a recommendation. The warning is a randomized matrix level with its own effect estimate.

## Reuse checklist

| Principle | Implementation |
|---|---|
| Six-agent team | `swarm.n_agents: 6` |
| Thirty action rounds | `swarm.max_rounds: 30` |
| Origin can only message | `origin_tools()` exposes only `dm_agent` |
| Five clean tasks | Each case has at least five tasks and validation enforces the count |
| Private memory | Per-agent memory tools and final snapshots |
| Final wipe warning | `CONTEXT_WIPE_PROMPT` and final memory phase |
| Four-action inbox cadence | `swarm.inbox_poll_interval: 4` in real-model manifests |
| Full and bridge graphs | `build_topology()` |
| Strict 0–3 score | Target rubrics and threshold 3 |
| Host-model sweep | Model is a matrix factor |
| Benign and harmful targets | Four supplied-paper target families |
| Warning defense | `mind_virus_warning` matrix factor |
| Source removal and reset | Persistence candidate manifest |
| Viral-theme taxonomy | `mechanisms.py` strategy outputs |
| Repeated trials | Seeded replicates and common randomization blocks |
| Complete transcripts | Append-only event trace and environment message log |
