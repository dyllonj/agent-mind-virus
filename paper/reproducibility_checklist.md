# Reproducibility checklist

- Archive the supplied paper source and record its checksum.
- Freeze the confirmatory YAML and record the validator fingerprint before inspecting outcomes.
- Record exact provider model identifiers and access dates.
- Retain base seed, generated run IDs, graph edges, origin identity, bridge identity, and execution order.
- Retain prompts, synthetic cases, target rubrics, tool schemas, and warning text.
- Report attempted, completed, failed, audited, excluded, and rerun counts.
- Publish run-level outcomes and enough redacted traces to audit the endpoint.
- Report judge prompts, raw judge outputs, aggregation rule, human-review sampling, and agreement.
- Keep fully connected runs out of the multi-hop denominator.
- Report both bootstrap and Newcombe intervals and the exact test.
- Report target-specific directions even if the aggregate result is positive.
- Label smoke and pilot outcomes correctly.
- List every deviation from the frozen manifest.
