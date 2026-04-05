# Audit: v2 Code (2026-04-05)

## Files Audited
- self_steer_v2.py (1185→1200 lines)
- two_model.py (455 lines)
- claude_steers_llama.py (531 lines)
- calibrate_features.py (692→693 lines)
- selfhost/client.py (112 lines)
- selfhost/server_direct.py (381→382 lines)

## Critical Issues Found — ALL FIXED

1. **server_direct.py:327** — inspect endpoint missing `global _current_interventions`. Fixed: added to global declaration + explicit clear.

2. **client.py:40 + self_steer_v2.py:744** — intervention `mode` field silently dropped in API payloads. `mode="clamp"` would silently become `mode="add"`. Fixed: both now include `"mode": i.get("mode", "add")`.

3. **calibrate_features.py** — 3 trials at temp 0.0 rotating over 2 prompts → trial 0 and 2 identical. Fixed: added 3rd test prompt.

## Moderate Issues — Noted, not all fixed

4. **self_steer_v2.py:535** — malformed tool call count inflated for minimal framing (short aliases match as substrings). Only affects the malformed metric, not actual parsing. Not fixed (low priority).

5. **client.py:78** — `inspect_full()` calls nonexistent `/v1/inspect_full` endpoint. Not called by any v2 script. Noted as trap.

6. **self_steer_v2.py:725** — parallel `chat_full()` bypasses SelfHostedClient.chat(). Bug fixes must be applied to both. Noted.

7. **SCENARIOS.md** — `--tool-tags` (XML-wrapped invisible tool calls) not implemented. Scaffold prompt says "write them inline" instead.

8. **PROPOSAL.md §12** — `--shuffled-labels` and `--mock-features` not implemented. Only `--opaque-labels` exists.

## Minor Issues — Fixed

9. **self_steer_v2.py** — added `--quick` flag (was missing, other scripts had it).
10. **self_steer_v2.py** — added server health check at startup.

## What's Clean

- Tool parsing consistent across all 3 experiment files (same regex patterns)
- Incremental saves on every round in all scripts
- Top-k sparsity (k=121) correctly implemented in SAE class
- Intervention range validation (0-65535) in all execution paths
- SAE hook correctly preserves reconstruction error
- All new flags (--opener, --conversation, --scaffold, --hint, --opaque-labels, --inspect-only) correctly wired through
- KeyboardInterrupt handling with state saves in two_model.py and claude_steers_llama.py
