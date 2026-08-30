# API cost plan

This planning snapshot uses the [standard global Claude API prices](https://platform.claude.com/docs/en/about-claude/pricing) checked on August 28, 2026. Claude Haiku 4.5 is priced at $1 per million input tokens and $5 per million output tokens. Claude Sonnet 4.6 is priced at $3 per million input tokens and $15 per million output tokens.

The local full-horizon rehearsal produced 5,920 host calls and 80 judge calls. Offline token counting over the exact traced prompts, histories, tools, tool calls, and visible responses estimated 18,029,021 host input tokens, 275,680 host output tokens, 106,646 judge input tokens, and 4,940 judge output tokens. This measured trace is the 1x profile. The 2x and 4x columns are transparent sensitivity cases in which all token volume is multiplied by two or four. They are not confidence intervals or hard caps.

| Stage | Rollouts | Maximum logical calls | 1x trace | 2x tokens | 4x tokens |
|---|---:|---:|---:|---:|---:|
| Paid matched canary | 2 | 750 | $2.48 | $4.95 | $9.90 |
| Weekend pilot | 16 | 6,000 | $19.80 | $39.60 | $79.21 |
| Confirmatory study | 144 | 54,000 | $178.21 | $356.43 | $712.85 |
| Persistence study | 48 | 16,560 | $54.68 | $109.37 | $218.73 |
| Defense study | 72 | 27,000 | $89.11 | $178.21 | $356.43 |
| Paper experiments, excluding canary | 280 | 103,560 | $341.80 | $683.61 | $1,367.22 |
| Total paper spend, including canary | 282 | 104,310 | $344.28 | $688.56 | $1,377.12 |

The separate 960-rollout factorial sweep is optional and overlaps with the lean paper design. Its projected cost is $1,188.09 at 1x, $2,376.18 at 2x, and $4,752.35 at 4x. Running it in addition to the lean paper would produce 1,242 paid rollouts and 464,310 maximum calls, with combined projected spend of $1,532.37 at 1x, $3,064.74 at 2x, or $6,129.47 at 4x.

These estimates assume every agent turn uses the maximum allowed tool-follow-up call, no provider retry is billed, prompt caching is disabled, and the paid conversations have the same mean token shape as the local deterministic trace after the stated multiplier. The estimates include model API usage only. Local computation and researcher time are not assigned a dollar price. Where the cost tooling renders a hard-budget verdict, it uses the nearest-rank p90 per-call token projection rather than the mean; the mean-based figures are retained for reference.

The 6,000 pilot figure is a ceiling on logical model calls, not a count of protocols. It consists of 5,920 host calls and 80 judge calls across 16 rollouts. The main paid manifests currently allow three retries, which means at most four provider attempts for one logical call. The validator therefore reports a pathological ceiling of 24,000 provider attempts for the pilot and 414,240 attempts across the 280 paper rollouts. Failed attempts are often not billed, but that depends on where a provider failure occurs. The 4x dollar column numerically covers four billed attempts with 1x-sized payloads; it does not cover four attempts whose payloads are also four times larger. The paid canary disables retries and therefore has the same 750 ceiling for logical calls and provider attempts.

The paid canary uses concurrency one and zero retries. Its provider-reported usage and cost must replace the local projection before the 16-rollout pilot is authorized. The same recalibration must happen again after the pilot before any confirmatory batch is frozen.

Prompt caching is not included in the budget. Anthropic's [prompt-caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) currently prices five-minute cache writes at 1.25 times base input cost and cache hits at 0.1 times base input cost. Haiku 4.5 requires at least 4,096 cacheable prompt tokens. The local trace shows that 1,359 of 5,920 host requests cross that threshold, so caching could help late in long conversations, but the realized saving depends on cache lifetime and scheduling. If caching is enabled, the canary must confirm nonzero cache-creation and cache-read fields and the manifest must be refingerprinted before scientific runs.

Batch pricing is also excluded. The host swarm is causally sequential across tool loops and communication rounds, while [Claude message batches](https://platform.claude.com/docs/en/api/http/messages/batches/create) may take up to 24 hours. Independent judge calls could be batched, but judge spend is only $6.94 in the 1x lean-paper estimate, so the maximum baseline saving there is small.
