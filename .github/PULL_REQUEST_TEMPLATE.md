# Description
<!-- What this PR does and why. Link repo issues with "Refs #N" / "Fixes #N".
     Never link private tracking tickets or chat sessions from a public repo. -->

# Type of change
<!-- Check one; delete the rest -->
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] New system or port (new OFS yaml, cards, fix-file set)
- [ ] Maintenance (refactor, cleanup, CI, comments)

# Change characteristics
<!-- Answer YES or NO on every line; one sentence of detail after a YES -->
- Changes outputs (values, COMOUT files or names, archives)?
- Breaking change to existing functionality?
- Requires a nos-utils submodule bump (nos-workflow) or a paired nos-workflow change (nos-utils)?
- Requires new or regenerated fix files on the machine?
- Requires a re-prep or a new hotstart before the next cycle?
- Machines affected: WCOSS2 / Hercules / both

# How has this been tested?
<!-- pytest counts, and the on-machine cycle (PDY/cyc, stages, result) or "not yet run" -->

# Checklist
- [ ] Dependent PRs merged and published (nos-utils before any submodule bump)
- [ ] pytest suite passes
- [ ] Existing operational behavior preserved, or the change is called out above
- [ ] No edits to a shared FIXofs while it is cycling
- [ ] Self-review done; yaml comments and docstrings updated where behavior changed
- [ ] No markdown files added to the repo (this template is the only exception)
