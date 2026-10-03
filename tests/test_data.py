"""Data-rule and loss tests."""
import collections

import pytest
import torch

from decision_model import data as D
from decision_model.losses import rps_sum, smooth_one_hot, soft_ce_sum
from decision_model.schema import Item


def mk(i, src="vitaminc", split="train", k=4, group=None, gold="a", target=None):
    keys = [chr(97 + j) for j in range(k)]
    crit = {kk: f"option {kk} of item {i}" for kk in keys}
    return Item(id=f"{src}:{i}", source=src, split=split, label_source="human", license="MIT", group=group or f"g{i}",
                state=f"state {i}", qtype="choice", instructions="which?", criteria=crit,
                target=target or {gold: 1.0}, gold=gold)


def mk_typed(qtype, criteria, gold, target=None, instructions="Is it so?"):
    return Item(id=f"x:{qtype}", source="x", split="train", label_source="human", license="MIT", group="g", state="s",
                qtype=qtype, instructions=instructions, criteria=criteria, target=target or {gold: 1.0}, gold=gold)


def test_excluded_sources_are_never_loaded():
    for bad in D.EXCLUDED:
        with pytest.raises(AssertionError):
            D.load_all_splits(sources=[bad])


def test_mixture_shares_and_epoch_cap():
    mix = {"vitaminc": 0.42, "snli": 0.30, "wanli_unanimous": 0.10, "intents": 0.15, "ai2_arc": 0.03}
    pools = {"vitaminc": [mk(i) for i in range(1000)], "snli": [mk(i, "snli") for i in range(1000)],
             "wanli_unanimous": [mk(i, "wanli") for i in range(1000)], "clinc_oos": [mk(i, "clinc") for i in range(500)],
             "massive": [mk(i, "massive") for i in range(300)], "banking77": [mk(i, "b77") for i in range(200)],
             "ai2_arc": [mk(i, "arc") for i in range(10)]}
    plan = D.sample_plan(pools, 10_000, seed=0, mix=mix)
    r = plan.report["sources"]
    assert r["ai2_arc"]["cap_binding"] and r["ai2_arc"]["sampled"] == D.MAX_EPOCHS * 10     # capped, not oversampled
    assert r["vitaminc"]["sampled"] == min(round(10_000 * 0.42), D.MAX_EPOCHS * 1000)   # its cap binds too
    assert max(collections.Counter(it.id for it in plan.items).values()) <= D.MAX_EPOCHS
    assert plan.digest() == D.sample_plan(pools, 10_000, seed=0, mix=mix).digest()          # deterministic


def test_disjointness_catches_shared_ids_and_groups():
    tr = [mk(1), mk(2)]
    with pytest.raises(AssertionError):
        D.assert_disjoint(tr, [mk(1, split="dev")])
    with pytest.raises(AssertionError):
        D.assert_disjoint(tr, [mk(9, split="dev", group="g2")])       # different id, same leak group
    D.assert_disjoint(tr, [mk(9, split="dev")])


class _Tok:
    """Whitespace tokenizer stand-in: 1 token per word."""
    def __call__(self, s, add_special_tokens=False):
        return type("E", (), {"input_ids": list(range(len(s.split())))})


def test_prefix_truncation_keeps_question_whole():
    it = Item(**{**mk(0).__dict__, "state": " ".join(["w"] * 5000), "instructions": "what is the right claim here"})
    ids, trunc = D.prefix_ids(_Tok(), it)
    assert trunc and len(ids) == D.MAX_PREFIX_TOKENS
    q_len = len(("\n\n" + it.instructions.strip()).split())
    assert ids[-q_len:] == list(range(q_len))                          # the question's tokens are the tail, intact


def test_option_texts_for_every_question_type():
    nl = mk_typed("yes_no", {"false": "Not toxic.", "true": "Toxic."}, "true")
    assert D.option_keys(nl) == ["false", "true"]
    assert [D.option_text(nl, k) for k in D.option_keys(nl)] == ["false: Not toxic.", "true: Toxic."]
    bare = mk_typed("yes_no", {}, "true", instructions="The sky is green.")
    assert [D.option_text(bare, k) for k in D.option_keys(bare)] == [
        "false: No. This is false: The sky is green.", "true: Yes. This is true: The sky is green."]
    sc = mk_typed("score", {"2": "high", "0": "low", "1": "mid"}, "1")
    assert D.option_keys(sc) == ["0", "1", "2"] and D.option_text(sc, "2") == "high"
    ch = mk_typed("choice", {"a": "Alpha", "b": ""}, "a")
    assert D.option_text(ch, "a") == "Alpha" and D.option_text(ch, "b") == "b"


def test_restricted_target_follows_the_key_order():
    it = mk(0, k=4, target={"a": 0.7, "b": 0.3}, gold="a")
    t = D.restricted_target(it, ["c", "b", "a", "d"])
    assert t == pytest.approx([0.0, 0.3, 0.7, 0.0])


def test_loss_ignores_rows_without_target_mass():
    lg = torch.randn(3, 4)
    tgt = torch.tensor([[1., 0, 0, 0], [0, 0, 0, 0], [0, .5, .5, 0]])
    s, n = soft_ce_sum(lg, tgt, torch.ones(3, 4, dtype=torch.bool))
    s2, n2 = soft_ce_sum(lg[[0, 2]], tgt[[0, 2]], torch.ones(2, 4, dtype=torch.bool))
    assert n == 2 and n2 == 2 and torch.allclose(s, s2)


def test_rps_is_zero_for_a_perfect_ordinal_prediction_and_grows_with_distance():
    m = torch.ones(1, 5, dtype=torch.bool)
    t = torch.tensor([[0, 0, 0, 1.0, 0]])
    near = rps_sum(torch.tensor([[0, 0, 20.0, 0, 0]]), t, m, torch.tensor([True]))
    far = rps_sum(torch.tensor([[20.0, 0, 0, 0, 0]]), t, m, torch.tensor([True]))
    exact = rps_sum(torch.tensor([[0, 0, 0, 20.0, 0]]), t, m, torch.tensor([True]))
    assert exact < 1e-6 < near < far
    assert rps_sum(torch.tensor([[20.0, 0, 0, 0, 0]]), t, m, torch.tensor([False])) == 0


def test_label_smoothing_touches_only_one_hot_rows():
    tgt = torch.tensor([[1.0, 0, 0, 0], [0.6, 0.4, 0, 0], [0, 1.0, 0, 0]])
    m = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    out = smooth_one_hot(tgt, m, 0.1)
    assert torch.allclose(out[0], torch.tensor([0.925, 0.025, 0.025, 0.025]))
    assert torch.equal(out[1], tgt[1])                                   # soft target untouched
    assert torch.allclose(out[2], torch.tensor([0.05, 0.95, 0, 0]))      # smoothing only over options read
    assert torch.allclose(out.sum(-1), torch.ones(3))


def test_typed_decisions_question_types_and_yes_no_fallback():
    from decision_model.sources import typed_decisions as TD
    yes_no = {"type": "other", "instructions": "Is it?"}
    assert TD.question_type(yes_no, {"probabilities": {"true": 0.7, "false": 0.3}, "label": "true"}) == "yes_no"
    assert TD.question_type({"type": "choice"}, {"probabilities": {"a": 1.0}, "label": "a"}) == "choice"
    assert TD.question_type({"type": "other"}, {"probabilities": {"a": 1.0}, "label": "a"}) is None
    assert set(TD.YES_NO_FALLBACK) == {"true", "false"}


def test_bare_yes_no_augmentation_adds_a_description_free_copy_of_every_yes_no_item():
    it = Item(id="x:1", source="x", split="train", label_source="human", license="-", group="g", state="s", qtype="yes_no",
              instructions="Is it urgent?", criteria={"false": "no", "true": "yes"}, target={"true": 1.0}, gold="true")
    (bare,) = D.AUGMENTATIONS["bare_yes_no"](it)
    assert bare.id == "x:1:bare" and bare.group == it.group and bare.gold == "true"
    assert D.option_text(bare, "true") == "true: Yes. This is true: Is it urgent?"
    choice = Item(**{**it.__dict__, "qtype": "choice", "criteria": {"a": "A", "b": "B"}, "target": {"a": 1.0}, "gold": "a"})
    assert D.AUGMENTATIONS["bare_yes_no"](choice) == []
