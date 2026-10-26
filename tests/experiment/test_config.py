import re

from experiment.clef_unseen_attack import config


def test_lora_targets_only_last_four_layers():
    matcher = re.compile(config.lora_target_regex((-4, -3, -2, -1), num_layers=32))
    assert matcher.fullmatch("model.language_model.layers.28.self_attn.q_proj")
    assert matcher.fullmatch("model.language_model.layers.31.self_attn.o_proj")
    assert matcher.fullmatch("model.language_model.layers.27.self_attn.q_proj") is None
    assert matcher.fullmatch("model.language_model.layers.30.mlp.down_proj") is None
    # PEFT on the bare text model sees names without a leading prefix.
    assert matcher.fullmatch("layers.31.linear_attn.in_proj_qkv")
    assert matcher.fullmatch("layers.27.linear_attn.in_proj_qkv") is None


def test_paths_are_inside_the_repo():
    # The clone may be named I-Spy-Agent or stegowatch; check the root, not its name.
    assert (config.ROOT / "pyproject.toml").exists()
    assert config.EXPERIMENT.name == "experiment"
    assert config.RESULTS.parent.name == "clef_unseen_attack"
