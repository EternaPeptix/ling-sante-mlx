# Rules
* Never run Python, builds or downloads on this Mac. Run everything with `LANE=<lane> scripts/r <cmd>` (zsh, never bash).
* Only touch 512S2 inside `~/science/src-<lane>`, `~/science/runs/<lane>`, and `~/science/models/<your converted dirs>`.
  `~/science/models/LingSante-3.1-flash-MXFP4` is READ-ONLY (the download writes there; wait for `DONE` in ~/science/dl.log
  before relying on all shards; you may read shards that already have an `OK` line).
* 512S2 also runs routerd tests in other lanes on CPU and a task-router service (port 18120): don't touch either.
* Commit in your worktree with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. If a command is denied, report it.
