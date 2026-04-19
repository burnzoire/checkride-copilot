Rewrite procedure steps in instructor voice using the ICP agent.

## Usage

`/rewrite-proc [key]`

- With a key (e.g. `/rewrite-proc agm_65f_missile_ir_seeker_only`): rewrite that procedure only.
- Without a key: rewrite all procedures in `data/airframes/fa18c/procedures/`.

## What to do

1. Read `data/airframes/fa18c/procedures/index.json` to find the target procedure(s).
2. For each procedure, read its JSON file.
3. For every step, use the **icp** subagent to rewrite `step["action"]` in instructor voice. Pass the raw action text and ask for a rewritten version only — no preamble, no explanation.
4. Set `step["voiced"]` to the rewritten text.
5. Write the updated JSON back to the same file (preserve all other fields exactly).
6. After each procedure, print a summary: procedure canonical name + how many steps were written.

**Strict 1:1 mapping required.** Every step gets a `voiced` field — including informational steps with no imperative action. For those, write a short check or confirm cue ("Confirm seeker is caged — holds boresight until uncaged.") rather than skipping or merging into the next step. The step number in `voiced` must always correspond to the same step number in `action`.

Show each rewritten step as you go so the user can spot problems before the file is saved. If a rewrite looks wrong (e.g. the control name was substituted, or it's longer than the original without good reason), flag it and ask before writing.
