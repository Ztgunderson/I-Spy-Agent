from experiment.clef_unseen_attack import corpus


def _action(**extra):
    return {
        "function": "text_editor",
        "arguments": {
            "command": "create",
            "path": "/tmp/a.py",
            "file_text": "print('hello')\n" * 5,
        },
        **extra,
    }


def _trajectory(label="honest", **extra):
    return {
        "trajectory_id": "t1",
        "run": "Haiku honest",
        "main_task": {"description": "Fix a test."},
        "side_task": None if label == "honest" else {"description": "attack"},
        "side_task_success": label == "success",
        "actions": [_action()],
        **extra,
    }


def test_cot_is_dropped_before_extraction():
    action = {
        "agent_reasoning": "hidden",
        "extended_thinking": "also hidden",
        "function": "text_editor",
    }
    out = corpus.drop_cot(action)
    assert out is action
    assert not (corpus.COT_FIELDS & out.keys())


def test_create_extracts_file():
    rows, stats = corpus.extract_trajectory(_trajectory())
    assert len(rows) == 1
    assert rows[0]["label"] == 0
    assert stats["create"] == 1


def test_failed_attack_is_unlabelled():
    rows, _ = corpus.extract_trajectory(_trajectory(label="failed"))
    assert rows[0]["label"] is None


def test_dedupe_runs_keeps_source_run_and_all_files():
    rows = [
        {"trajectory_id": "t1", "run": "Haiku attack", "content": "a"},
        {"trajectory_id": "t1", "run": "Haiku attack", "content": "b"},
        {"trajectory_id": "t1", "run": "GPT-5 Nano monitor on Haiku attack", "content": "a"},
    ]
    out = corpus.dedupe_runs(rows)
    assert len(out) == 2
    assert {row["content"] for row in out} == {"a", "b"}
    assert all(row["run"] == "Haiku attack" for row in out)
