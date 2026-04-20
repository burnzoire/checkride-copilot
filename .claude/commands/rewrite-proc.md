Rewrite procedure steps in instructor voice using the ICP agent.

## Usage

`/rewrite-proc [key]`

- With a key (e.g. `/rewrite-proc agm_65f_missile_ir_seeker_only`): rewrite that procedure only.
- Without a key: rewrite all procedures in `data/airframes/fa18c/procedures/`.

## What to do

1. Read `data/airframes/fa18c/procedures/index.json` to find the target procedure(s).
2. For each procedure, read its JSON file.
3. **Pacing pass first** — before rewriting any text, read through all steps and decide `pace` for each:
   - `"checklist"` — discrete action the pilot can pause before and after
   - `"maneuver"` — part of a continuous physical maneuver (no realistic pause point)
   
   Assign `pace` based on the ICP agent's pacing rules. Print your pacing decisions as a table (step num | pace | reason) and ask for confirmation before proceeding to the rewrite pass. If any grouping looks wrong, resolve it before writing.

4. **Rewrite pass** — for every step, use the **icp** subagent to rewrite `step["action"]` in instructor voice according to its pace:
   - `checklist` steps: one clean imperative sentence, ≤20 words
   - `maneuver` steps: short, urgent, flows into the next — write as if calling it on hot mic mid-maneuver

5. Set both `step["voiced"]` and `step["pace"]` on every step. Do not remove or alter any other fields.

6. Write the updated JSON back to the same file (preserve all other fields exactly).

7. After each procedure, print a summary: canonical name, step count, how many were marked maneuver.

**Strict 1:1 mapping required.** Every step gets both `voiced` and `pace` — including informational steps (assign `checklist`, write a short confirm cue). Never skip or merge steps in the JSON structure — grouping for delivery is handled at runtime by the `pace` field.

Show each rewritten step as you go so the user can spot problems before the file is saved. If a rewrite looks wrong (control name substituted, longer than original without good reason, or pace seems wrong), flag it and ask before writing.
