"""BoolQ adapter: item construction on hand-made rows (offline)."""
import pandas as pd

from decision_model.schema import validate
from decision_model.sources import boolq as B
from decision_model.sources._common import Report


def frame(rows):
    return pd.DataFrame(rows, columns=["question", "answer", "passage"])


def test_question_text_is_capitalised_with_question_mark():
    assert B.question_text("is the sky blue") == "Is the sky blue?"
    assert B.question_text("  is it?  ") == "Is it?"
    assert B.question_text("") == ""


def test_items_are_valid_yes_no_with_one_hot_targets():
    rep = Report("boolq")
    rows = B.rows_from_frame(frame([("is water wet", True, "Water is a liquid. It wets things."),
                                    ("is fire cold", False, "Fire is hot. It burns.")]), "train", rep)
    B.assign_groups(rows)
    its = B.items(rows, "train")
    for it in its:
        validate(it)
        assert it.qtype == "yes_no" and it.criteria == B.CRITERIA and it.label_source == "human" and it.license == "CC-BY-SA-3.0"
    assert [it.gold for it in its] == ["true", "false"]
    assert its[0].target == {"true": 1.0} and its[1].target == {"false": 1.0}
    assert its[0].instructions == "Is water wet?" and its[0].state == "Water is a liquid. It wets things."


def test_conflicting_and_exact_duplicates_are_dropped_and_counted():
    rep = Report("boolq")
    rows = B.rows_from_frame(frame([("q one", True, "P one."), ("q one", False, "P one."),
                                    ("q two", True, "P two."), ("q two", True, "P two."),
                                    ("", True, "P three.")]), "train", rep)
    assert [r["question"] for r in rows] == ["Q two?"]
    assert rep.drops["train:conflicting_duplicate"] == 2
    assert rep.drops["train:exact_duplicate"] == 1
    assert rep.drops["train:empty_text"] == 1


def test_questions_on_the_same_passage_share_a_group():
    rep = Report("boolq")
    rows = B.rows_from_frame(frame([("q a", True, "Same passage here. More text."),
                                    ("q b", False, "Same passage here. More text."),
                                    ("q c", True, "Same passage here. Different continuation."),
                                    ("q d", True, "Unrelated passage. Text.")]), "train", rep)
    B.assign_groups(rows)
    g = [r["group"] for r in rows]
    assert g[0] == g[1] == g[2] and g[3] != g[0]              # same passage, and same opening sentence, merge
