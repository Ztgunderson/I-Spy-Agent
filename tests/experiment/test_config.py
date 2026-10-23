import re

from experiment.clef_unseen_attack import config


def test_lora_targets_only_last_four_layers():
    matcher = re.compile(config.lora_target_regex((-4, -3, -2, -1), num_layers=32))
    assert matcher.fullmatch("model.language_model.layers.28.self_attn.q_proj")
    assert matcher.fullmatch("model.language_model.layers.31.self_attn.o_proj")
    assert matcher.fullmatch("model.language_model.layers.27.self_attn.q_proj") is None
    assert matcher.fullmatch("model.language_model.layers.30.mlp.down_proj") is None


def test_paths_are_inside_i_spy_agent():
    assert config.ROOT.name == "I-Spy-Agent"
    assert config.EXPERIMENT.name == "experiment"
    assert config.RESULTS.parent.name == "clef_unseen_attack"
