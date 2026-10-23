"""Extraction of (file content, stated purpose) pairs from linuxarena-public.

Schema verified against the real corpus (120 trajectories scanned, 2026-10-16):

  bash         -> {"cmd": str}                                              (5451 calls / 120 traj)
  bash         -> {"cmd": str, "timeout": int}
  text_editor  -> {"command": "view",        "path": str}                  (1701)
  text_editor  -> {"command": "str_replace", "old_str", "new_str", "path"} (978)
  text_editor  -> {"command": "create",      "file_text": str, "path": str} (488)
  submit       -> {} or {"answer"/"confusions": ...}

Only `create` yields a complete file body. `str_replace` is a patch and needs prior state.
`bash` mutates files in place, so recovering its result means replaying the command sequence
in a sandbox -- that is a Docker-scale job and is deliberately NOT attempted here.

Chain-of-thought fields (`agent_reasoning`, `extended_thinking`) exist in this corpus and are
stripped on read. They are attacker-controlled; a pipeline that reads them is not testing the
hypothesis under test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Deliberately removed before anything downstream can see an action.
COT_FIELDS = frozenset({"agent_reasoning", "extended_thinking"})

# Matches: cat > path <<'EOF' ... EOF   |   cat >> path <<-EOF ... EOF   |  tee path <<EOF ... EOF
HEREDOC = re.compile(
    r"(?:cat\s+>?\s*|tee\s+(?:-\w+\s+)*)([\w./-]+)\s*<<-?\s*['\"]?(\w+)['\"]?\n(.*?)\n\2",
    re.S,
)

MIN_CONTENT_CHARS = 40


def drop_cot(action: dict) -> dict:
    """Strip chain-of-thought from an action in place. Raises if anything survives."""
    for f in COT_FIELDS:
        action.pop(f, None)
    leaked = COT_FIELDS & action.keys()
    if leaked:  # pragma: no cover - defensive
        raise AssertionError(f"chain-of-thought leaked: {leaked}")
    return action


def parse_heredocs(cmd: str) -> list[dict]:
    return [{"path": m.group(1), "content": m.group(3)} for m in HEREDOC.finditer(cmd)]


def extract_trajectory(traj: dict) -> tuple[list[dict], dict]:
    """Return (file rows, tier stats) for one trajectory.

    Label semantics (deliberately conservative):
      0  -> honest run (trajectory has no side task). These are true negatives.
      1  -> attack run AND the side task actually scored. Only these are positives.
      None-> attack run that failed, or unscored. Never used as a negative: labelling a
            failed attack's ordinary work as benign would teach the model to ignore
            fragments that only became dangerous in a run that didn't land.
    """
    purpose = (traj.get("main_task") or {}).get("description") or ""
    honest = traj.get("side_task") is None
    success = bool(traj.get("side_task_success"))
    label = 0 if honest else (1 if success else None)

    rows: list[dict] = []
    stats = {"create": 0, "str_replace_skipped": 0, "heredoc": 0, "bash_calls": 0}

    def emit(path, content, tier):
        if not isinstance(content, str) or not path:
            return
        if len(content.strip()) < MIN_CONTENT_CHARS:
            return
        stats[tier] = stats.get(tier, 0) + 1
        rows.append(
            {
                "trajectory_id": traj.get("trajectory_id"),
                "run": traj.get("run_display_name"),
                "agent_model": traj.get("agent_model"),
                "purpose": purpose,
                "path": path,
                "content": content,
                "tier": tier,
                "label": label,
            }
        )

    for a in traj.get("actions", []):
        drop_cot(a)
        fn = a.get("function")
        args = a.get("arguments") or {}
        if not isinstance(args, dict):
            continue

        if fn == "text_editor":
            op = args.get("command")
            if op == "create":
                emit(args.get("path"), args.get("file_text"), "create")
            elif op == "str_replace":
                # Patch only. Emitting old_str/new_str as if it were a file body would
                # produce training data that is not a file.
                stats["str_replace_skipped"] += 1
        elif fn == "bash":
            stats["bash_calls"] += 1
            cmd = args.get("cmd") or args.get("command") or ""
            for h in parse_heredocs(cmd):
                emit(h["path"], h["content"], "heredoc")

    return rows, stats


def build_corpus(sample_path: Path, limit: int | None = None, progress_every: int = 200):
    rows: list[dict] = []
    totals: dict[str, int] = {}
    n = 0
    with sample_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r, s = extract_trajectory(json.loads(line))
            rows.extend(r)
            for k, v in s.items():
                totals[k] = totals.get(k, 0) + v
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"  {n} trajectories -> {len(rows)} files", flush=True)
            if limit and n >= limit:
                break
    return rows, n, totals


def dedupe(rows: list[dict]) -> list[dict]:
    """Identical files recur across trajectories; keep first occurrence."""
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r["path"], hash(r["content"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def dedupe_by_trajectory(rows: list[dict]) -> list[dict]:
    """Drop monitor re-runs of the same trajectory.

    linuxarena-public ships several runs that are the SAME trajectories scored by a
    different monitor -- e.g. "Haiku honest" (200) reappears as "GPT-5 Nano monitor on
    Opus honest" (200) and "Opus monitor on Opus honest" (200). The dataset card is
    explicit that only monitor_response / traj_monitor_response differ between them.

    Counting those separately would silently triple-count every honest file and inflate
    the negative set. Keep one row per trajectory_id, preferring a source agent run
    over a monitor re-run of that same trajectory.
    """
    by_traj: dict[str, dict] = {}
    for r in rows:
        tid = r.get("trajectory_id")
        if tid is None:
            continue
        cur = by_traj.get(tid)
        if cur is None:
            by_traj[tid] = r
            continue
        cur_is_rerun = is_monitor_rerun(cur.get("run"))
        if cur_is_rerun and not is_monitor_rerun(r.get("run")):
            by_traj[tid] = r
    return list(by_traj.values())


def is_monitor_rerun(run) -> bool:
    return "monitor on" in str(run or "").lower()


def dedupe_runs(rows: list[dict]) -> list[dict]:
    """Keep one run per trajectory, preserving every file in that run.

    linuxarena-public ships the same trajectory under several monitor-run
    names. ``dedupe_by_trajectory`` keeps only a single file, which loses real
    files; this function picks the source (non-monitor) run for each
    trajectory and keeps all of its rows instead.

    Rows without a trajectory id are kept once per content hash.
    """
    best_run: dict[str, object] = {}
    for row in rows:
        tid = row.get("trajectory_id")
        if tid is None:
            continue
        run = row.get("run")
        current = best_run.get(tid)
        if current is None or (is_monitor_rerun(current) and not is_monitor_rerun(run)):
            best_run[tid] = run

    out: list[dict] = []
    seen_anonymous: set[int] = set()
    for row in rows:
        tid = row.get("trajectory_id")
        if tid is None:
            key = hash(row.get("content", ""))
            if key in seen_anonymous:
                continue
            seen_anonymous.add(key)
            out.append(row)
        elif row.get("run") == best_run[tid]:
            out.append(row)
    return out