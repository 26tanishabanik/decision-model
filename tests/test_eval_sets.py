"""Metric harness on hand-worked cases, and checks on the built frozen test sets."""
import hashlib
import json
import math

import numpy as np
import pytest

from decision_model.evaluate import (
    EVAL_DIR,
    acc_at_coverage,
    aurc,
    compare,
    coverage_curve,
    ece,
    per_item,
    summarize,
    top_share,
)
from decision_model.schema import Item, read_jsonl, validate

ABCD = {"A": "option a", "B": "option b", "C": "option c", "D": "option d"}


def mk(i, gold="A", roles=None, crit=None, qtype="choice", target=None, meta=None):
    crit = crit or dict(ABCD)
    return Item(id=f"t:{i}", source="t", split="eval", label_source="human", license="MIT", group=f"g{i}", state="some state",
                qtype=qtype, instructions="Which?", criteria=crit, target=target or {gold: 1.0}, gold=gold,
                roles=roles or {}, meta=meta or {})


def pr(item, probs):
    return {"id": item.id, "probs": probs}


# ---- primitives ------------------------------------------------------------------------------------------------------
def test_top_share_strict_and_ties():
    assert top_share(0.5, [0.3, 0.2]) == 1.0
    assert top_share(0.3, [0.5, 0.2]) == 0.0
    assert top_share(0.25, [0.25, 0.25, 0.25]) == pytest.approx(0.25)
    assert top_share(0.4, [0.4, 0.2]) == pytest.approx(0.5)
    assert top_share(0.1, []) == 1.0


def test_ece_tiny_example():
    # bins of width 1/15: 0.9 -> (0.867,0.933] ; 0.6 -> its own bin ; 0.3 -> (0.267,0.3]
    # |0.5-0.9|*2 + |1-0.6|*1 + |0-0.3|*1 = 0.8 + 0.4 + 0.3 = 1.5 ; / 4 = 0.375
    assert ece([0.9, 0.9, 0.6, 0.3], [1, 0, 1, 0]) == pytest.approx(0.375)
    assert ece([0.7, 0.7], [1, 0]) == pytest.approx(0.2)
    assert ece([1.0, 1.0], [1, 1]) == pytest.approx(0.0)


def test_coverage_curve_aurc_and_ties():
    conf, cor = [0.9, 0.8, 0.7, 0.6], [1, 0, 1, 1]
    assert np.allclose(coverage_curve(conf, cor), [1, 0.5, 2 / 3, 0.75])
    assert aurc(conf, cor) == pytest.approx((0 + 0.5 + 1 / 3 + 0.25) / 4)
    assert acc_at_coverage(conf, cor, 0.5) == pytest.approx(0.5)     # k = 2
    assert acc_at_coverage(conf, cor, 0.8) == pytest.approx(0.75)    # k = ceil(3.2) = 4
    assert acc_at_coverage(conf, cor, 1.0) == pytest.approx(0.75)
    # a tied block is averaged, so input order does not matter
    assert np.allclose(coverage_curve([0.9, 0.9, 0.5], [1, 0, 1]), [0.5, 0.5, 2 / 3])
    assert np.allclose(coverage_curve([0.9, 0.9, 0.5], [0, 1, 1]), [0.5, 0.5, 2 / 3])


# ---- roles-based metrics on a hand-worked 4-item set -----------------------------------------------------------------
def roles_fixture():
    r_full = {"A": "correct", "B": "hard", "C": "unrelated", "D": "unrelated"}
    items = [mk(1, roles=r_full), mk(2, roles=r_full), mk(3, roles=r_full),
             mk(4, roles={"A": "correct", "B": "unrelated", "C": "unrelated", "D": "unrelated"})]
    P = [{"A": .4, "B": .3, "C": .2, "D": .1},     # right; above hard and unrelated
         {"A": .3, "B": .5, "C": .1, "D": .1},     # hard on top: relevance 1, relational 0
         {"A": .2, "B": .1, "C": .6, "D": .1},     # unrelated on top: relevance 0, relational 1
         {"A": .25, "B": .25, "C": .25, "D": .25}]  # 4-way tie: 1/4 everywhere; no hard negative
    return items, {it.id: pr(it, p) for it, p in zip(items, P)}, P


def test_roles_relevance_relational_full_decision():
    items, preds, P = roles_fixture()
    r = per_item(items, preds)
    assert np.allclose(r["relevance"], [1, 1, 0, 0.25])
    assert np.allclose(r["relational"][:3], [1, 0, 1]) and np.isnan(r["relational"][3])
    assert np.allclose(r["full"], [1, 0, 0, 0.25])
    s = summarize(items, preds, n_boot=50)
    ro = s["roles"]
    assert ro["relevance"]["point"] == pytest.approx(2.25 / 4)
    assert ro["relational"]["point"] == pytest.approx(2 / 3) and ro["n_relational"] == 3
    assert ro["full_decision"]["point"] == pytest.approx(1.25 / 4)
    assert s["accuracy"]["point"] == pytest.approx(1.25 / 4)
    # hard-prediction errors: item 2 (B = hard), item 3 (C = unrelated); item 4's tie breaks to A = gold
    assert ro["n_errors"] == 2
    assert ro["hard_on_top_rate"] == pytest.approx(0.5)
    assert ro["unrelated_on_top_rate"] == pytest.approx(0.5)


def test_roles_standardized_margins():
    items, preds, P = roles_fixture()
    r = per_item(items, preds)
    sd = np.std(np.log(np.array([list(p.values()) for p in P])).ravel())
    assert r["m_full"][0] == pytest.approx((math.log(.4) - math.log(.3)) / sd)
    assert r["m_relational"][1] == pytest.approx((math.log(.3) - math.log(.5)) / sd)
    assert r["m_relevance"][2] == pytest.approx((math.log(.2) - math.log(.6)) / sd)
    assert r["m_full"][3] == pytest.approx(0.0)


def test_logloss_brier_ece_on_roles_fixture():
    items, preds, P = roles_fixture()
    s = summarize(items, preds, n_boot=50)
    assert s["log_loss"]["point"] == pytest.approx(-(math.log(.4) + math.log(.3) + math.log(.2) + math.log(.25)) / 4)
    # Brier (sum over options): .50 + .76 + 1.02 + .75
    assert s["brier"]["point"] == pytest.approx(3.03 / 4)
    # conf = max p: .4 (c=1), .5 (c=0), .6 (c=0), .25 (c=.25), each alone in its bin
    assert s["ece"]["point"] == pytest.approx((0.6 + 0.5 + 0.6 + 0.0) / 4)


def test_soft_target_kl_and_cross_entropy():
    it = mk(1, gold="a", crit={"a": "x", "b": "y"}, target={"a": .7, "b": .3})
    r = per_item([it], {it.id: pr(it, {"a": .7, "b": .3})})
    h = -(.7 * math.log(.7) + .3 * math.log(.3))
    assert r["ce"][0] == pytest.approx(h) and r["kl"][0] == pytest.approx(0, abs=1e-12)
    assert r["brier"][0] == pytest.approx(0)


def test_macro_f1_only_for_class_label_sets():
    crit = {"false": "no", "true": "yes"}
    items = [mk(i, gold=g, crit=crit, qtype="yes_no", meta={"f1_space": ""})
             for i, g in enumerate(["true", "true", "false", "false"])]
    preds = {it.id: pr(it, {"true": p, "false": 1 - p}) for it, p in zip(items, [.9, .2, .1, .3])}
    s = summarize(items, preds, n_boot=50)
    # true: P=1 R=.5 F1=2/3 ; false: P=2/3 R=1 F1=.8
    assert s["macro_f1"]["point"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert s["accuracy"]["point"] == pytest.approx(0.75)
    items2 = [mk(i) for i in range(3)]
    assert summarize(items2, {it.id: pr(it, {"A": 1}) for it in items2}, n_boot=20)["macro_f1"] is None


def test_abstention_ranks_by_the_top_probability():
    items = [mk(i) for i in range(4)]
    probs = [{"A": .1, "B": .9}, {"A": .6, "B": .4}, {"A": .7, "B": .3}, {"A": .8, "B": .2}]
    s = summarize(items, {it.id: pr(it, p) for it, p in zip(items, probs)}, n_boot=20)
    assert s["abstention"]["confidence"] == "max_p"
    assert s["abstention"]["acc@50"] == pytest.approx(0.5) and s["abstention"]["acc@100"] == pytest.approx(0.75)


def test_prediction_validation():
    it = mk(1)
    with pytest.raises(ValueError):
        per_item([it], {})
    with pytest.raises(ValueError):
        per_item([it], {it.id: pr(it, {"Z": 1.0})})
    with pytest.raises(ValueError):
        per_item([it], {it.id: pr(it, {"A": -1.0, "B": 2.0})})
    r = per_item([it], {}, allow_missing=True)
    assert r["n_missing"] == 1 and r["correct"][0] == pytest.approx(0.25)
    r = per_item([it], {it.id: pr(it, {"A": 2.0, "B": 2.0})})    # renormalised; missing keys -> 0
    assert r["correct"][0] == pytest.approx(0.5) and r["brier"][0] == pytest.approx(0.25 + 0.25)


def test_paired_compare_mcnemar_and_diff():
    items = [mk(i) for i in range(10)]
    a = {it.id: pr(it, {"A": .9, "B": .1}) for it in items}
    b = {it.id: pr(it, {"A": .1, "B": .9}) if i < 6 else pr(it, {"A": .9, "B": .1}) for i, it in enumerate(items)}
    c = compare(items, a, b, n_boot=200)
    assert c["mcnemar"]["a_only_correct"] == 6 and c["mcnemar"]["b_only_correct"] == 0
    assert c["mcnemar"]["p"] == pytest.approx(2 * 0.5 ** 6)
    assert c["accuracy"]["diff"] == pytest.approx(0.6)
    assert c["accuracy"]["lo"] > 0


# ---- frozen eval files ---------------------------------------------------------------------------------------------
MANIFEST = EVAL_DIR / "manifest.json"
needs_eval = pytest.mark.skipif(not any(EVAL_DIR.glob("*.jsonl")), reason="eval sets not built (scripts/build_eval_sets.py)")


@needs_eval
def test_eval_files_validate_unique_ids_and_match_manifest():
    man = json.loads(MANIFEST.read_text())["sets"]
    seen = set()
    for name, entry in man.items():
        path = EVAL_DIR / f"{name}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], name
        ids = []
        for it in read_jsonl(path):
            validate(it)
            assert it.split == "eval", it.id
            if it.roles:
                assert [k for k, r in it.roles.items() if r == "correct"] == [it.gold], it.id
            ids.append(it.id)
        assert len(ids) == entry["count"] and len(set(ids)) == len(ids), name
        assert not (seen & set(ids)), name
        seen |= set(ids)
        assert entry["revision"], name


@needs_eval
def test_eval_set_label_sources_and_option_counts():
    man = json.loads(MANIFEST.read_text())["sets"]
    assert man["typed_decisions_test"]["label_source"] == ["llm"]
    assert all(man[n]["label_source"] == ["human"] for n in man if n != "typed_decisions_test")
    assert man["scifact_select"]["roles"].keys() == {"correct", "hard", "unrelated"}
    assert man["banking77_test"]["n_options"]["min"] == 77 and man["clinc_test"]["n_options"]["min"] == 151
    assert man["clinc_test"]["gold_top5"][0] == ["out_of_scope", 1000]
