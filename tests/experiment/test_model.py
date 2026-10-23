import torch

from experiment.clef_unseen_attack.model import (
    OptionLexicon,
    _text_backbone,
    file_state,
    schema_hash,
)


class _Inner:
    language_model = "language-model"
    other = "model"


class _Base:
    model = _Inner()
    language_model = "fallback"


def test_text_backbone_prefers_language_model():
    assert _text_backbone(_Base()) == "language-model"


def test_option_lexicon_slices_and_rejects_unknown():
    weight = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    lexicon = OptionLexicon.from_weight(weight, [1, 2])
    rows = lexicon[torch.tensor([1, 2])]
    assert rows.shape == (2, 4)
    assert torch.equal(rows[0], weight[1])
    try:
        lexicon[torch.tensor([9])]
    except KeyError:
        pass
    else:
        raise AssertionError("unknown option token should raise")


def test_schema_hash_is_stable_and_sensitive():
    first = schema_hash({"covert": {"type": "noul"}})
    second = schema_hash({"covert": {"type": "noul"}})
    third = schema_hash({"covert": {"type": "choice"}})
    assert first == second
    assert first != third


def test_file_state_matches_stage0_composition():
    class Record:
        purpose = "purpose"
        content = "content"

    assert file_state(Record()) == "PURPOSE: purpose\n---\ncontent"
