# Safety and scope

This repository is a closed evaluation harness. It studies overt natural-language influence among synthetic agents in a local simulator. Agents cannot access a shell, execute code, browse the network, use credentials, contact people, publish content, purchase anything, or modify source evidence. Direct messages exist only inside one rollout and only along configured graph edges.

Some target positions are intentionally objectionable because the supplied paper compares benign and misaligned ideologies. They are synthetic experimental stimuli, not endorsed claims. Raw traces should be handled as adversarial content and should not be inserted into operational agent memory, production prompts, or public autonomous systems.

The mock backend deliberately produces a cascade so that software paths can be tested. Its outcomes are fixtures, not evidence. The real-model results apply only to the exact model versions, prompts, cases, graph, tool affordances, and sampling settings recorded in the manifest.

The harness does not optimize a transferable payload, deploy one outside the simulator, or give agents tools for real-world propagation. Adding external messaging, real credentials, public posting, self-modifying code, or uncontrolled network access would materially expand scope and requires a separate safety review.
