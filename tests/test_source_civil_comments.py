"""Civil Comments adapter: item construction on hand-made rows (no network)."""
import pytest

from decision_model.eval_sets import TOXICITY_CRIT, TOXICITY_Q
from decision_model.schema import validate
from decision_model.sources import civil_comments as C


def row(t, i=0, text="you are an idiot"):
    return {"id": f"train-x-{i}", "text": text, "toxicity": t, "group": "cc-g:abc", "native_split": "train"}


@pytest.mark.parametrize("t,gold,target", [
    (0.0, "false", {"false": 1.0}),
    (1.0, "true", {"true": 1.0}),
    (0.2, "false", {"true": 0.2, "false": 0.8}),
    (0.5, "true", {"true": 0.5, "false": 0.5}),
    (0.7, "true", {"true": 0.7, "false": pytest.approx(0.3)}),
])
def test_soft_target_and_gold(t, gold, target):
    it = C.make_item(row(t), "train")
    validate(it)
    assert it.qtype == "yes_no" and it.gold == gold and it.target == target and it.label_source == "human_multi"
    assert it.license == "CC0-1.0" and it.meta["toxicity_fraction"] == t


def test_wording_is_ours_not_the_eval_wording():
    assert TOXICITY_Q not in C.TEMPLATES
    assert set(C.CRITERIA.values()).isdisjoint(set(TOXICITY_CRIT.values()))
    assert set(C.CRITERIA) == {"false", "true"}


def test_template_choice_is_stable_and_uses_all_templates():
    ids = [f"civil_comments:train-x-{i}" for i in range(300)]
    first = [C.template_for(i) for i in ids]
    assert first == [C.template_for(i) for i in ids]
    assert set(first) == set(C.TEMPLATES)


def test_out_of_range_fraction_rejected():
    with pytest.raises(AssertionError):
        C.target_of(1.2)
