"""Unit tests for the quality gates. Items here are hand-written test fixtures, not training data."""
from decision_model.quality import (
    NearDupIndex,
    add_unrelated_negatives,
    dedup_against_eval,
    gold_position_stats,
    jaccard,
    shingles,
)
from decision_model.schema import Item, validate


def mk(i, state, instr="Which is correct?", crit=None, gold="a", group=None, roles=None, split="train"):
    crit = crit or {"a": "first option text", "b": "second option text"}
    return Item(id=f"t:{i}", source="t", split=split, label_source="human", license="MIT", group=group or f"g{i}", state=state,
                qtype="choice", instructions=instr, criteria=crit, target={gold: 1.0}, gold=gold, roles=roles or {})


def test_shingle_jaccard_basic():
    a = shingles("the quick brown fox jumps over the lazy dog")
    assert jaccard(a, a) == 1.0
    assert jaccard(a, shingles("completely different sentence about cats and dogs here")) == 0.0


def test_near_dup_index_finds_light_edit_not_unrelated():
    idx = NearDupIndex(0.8)
    base = "The committee approved the new budget for the city library after a long debate on Tuesday evening in the hall"
    idx.add("e1", base)
    assert idx.query(base.replace("Tuesday", "Tuesday,"))  # punctuation-only change -> identical after normalization
    assert not idx.query("A cat sat on the warm windowsill watching birds in the garden all afternoon long today")


def test_dedup_removes_train_item_matching_eval():
    ev = [mk("e", "Scientists found that the drug reduced mortality in elderly patients with heart failure", split="eval")]
    tr = [mk(1, "Scientists found that the drug reduced mortality in elderly patients with heart failure"),
          mk(2, "An unrelated sentence about football results from the weekend match in Madrid and Barcelona")]
    kept, removed = dedup_against_eval(tr, ev)
    assert [k.id for k in kept] == ["t:2"] and removed[0][0] == "t:1"


def test_gold_position_imbalance_flag():
    items = [mk(i, f"state {i}", gold="a") for i in range(20)]
    s = gold_position_stats(items)
    assert s["2"]["max_position_share"] == 1.0 and s["2"]["imbalanced"]


def test_unrelated_negatives_from_other_groups_only():
    items = []
    for i in range(10):
        crit = {"a": f"correct answer number {i} about topic {i}", "b": f"wrong answer number {i} about topic {i}"}
        items.append(mk(i, f"state text {i}", crit=crit, roles={"a": "correct", "b": "hard"}))
    out, rep = add_unrelated_negatives(items, n_neg=2)
    assert rep["augmented"] == 10
    for it in out:
        validate(it)
        added = [k for k, r in it.roles.items() if r == "unrelated"]
        assert len(added) == 2
        assert all(it.criteria[k] != it.criteria["a"] for k in added)
        assert all(f"topic {it.id.split(':')[1]} " not in it.criteria[k] + " " for k in added)


def test_same_answer_catches_plural_and_containment():
    from decision_model.quality import same_answer
    assert same_answer("cell", "Cells")
    assert same_answer("waste bin", "the waste bin")
    assert not same_answer("japan", "france")
