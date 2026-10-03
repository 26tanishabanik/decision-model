"""HelpSteer2 adapter rules, on hand-made rows"""
from decision_model.schema import validate
from decision_model.sources import helpsteer2 as H
from decision_model.sources._common import Report


def row(i, prompt="What is 2+2?", response="4.", split="train", **labels):
    lab = {a: 3 for a in H.ATTRIBUTES}
    lab.update(labels)
    return {"prompt": prompt, "response": response, **lab, "row": i, "native_split": split,
            "group": f"hs2-g:{H.text_hash(prompt)}"}


def test_five_score_items_per_row_with_ordered_levels():
    items = H.make_items([row(0)], {}, "train", Report("t"))
    assert [it.meta["attribute"] for it in items] == list(H.ATTRIBUTES)
    for it in items:
        validate(it)
        assert it.qtype == "score" and list(it.criteria) == ["0", "1", "2", "3", "4"]
        assert it.gold == "3" and it.target == {"3": 1.0} and it.label_source == "human"


def test_soft_target_from_individual_ratings():
    t, kind = H.target_for([3, 3, 4], 3)
    assert kind == "soft" and t == {"3": 2 / 3, "4": 1 / 3}


def test_released_label_not_an_argmax_falls_back_to_one_hot():
    t, kind = H.target_for([2, 2, 4], 3)          # released = rounded mean 2.67 -> 3, not a mode
    assert kind == "onehot_fallback" and t == {"3": 1.0}


def test_tied_modes_including_released_label_stay_soft_and_valid():
    ann = {("p", "r"): {a: [2, 3, 4] for a in H.ATTRIBUTES}}
    items = H.make_items([row(0, "p", "r")], ann, "train", Report("t"))
    for it in items:
        validate(it)
        assert it.label_source == "human_multi" and it.meta["target_kind"] == "soft" and len(it.target) == 3


def test_group_is_the_prompt_so_both_responses_share_it():
    a, b = row(0, "Same prompt", "answer one"), row(1, "same  prompt!", "answer two")
    assert a["group"] == b["group"]
    its = H.make_items([a, b], {}, "train", Report("t"))
    assert len({it.group for it in its}) == 1


def test_multi_turn_markers_are_rendered_as_roles():
    s = H.state_text("hi\n<extra_id_1>Assistant\nHello!\n<extra_id_1>User\nlist panels", "Panel, FlowLayoutPanel")
    assert "<extra_id_1>" not in s
    assert s.startswith("User request:\nUser: hi\n\nAssistant: Hello!\n\nUser: list panels")
    assert s.endswith("Assistant response:\nPanel, FlowLayoutPanel")


def test_newlines_and_indentation_inside_code_are_kept():
    assert H.keep_lines("def f():\n    return  1   \n\n\n\nx") == "def f():\n    return 1\n\nx"


def test_conflicting_duplicate_rows_are_dropped():
    rep = Report("t")
    out = H.dedup_rows([row(0, helpfulness=1), row(1, helpfulness=4), row(2, "other")], rep, "train")
    assert [r["prompt"] for r in out] == ["other"] and rep.drops["train:conflicting_duplicate_rows"] == 2


def test_over_long_state_is_dropped():
    rep = Report("t")
    assert H.make_items([row(0, response="w " * 3100)], {}, "train", rep) == []
    assert rep.drops["train:state_over_3000_words"] == 5
