"""Shared source helpers."""
from decision_model.sources._common import Report, nli_items


def test_premise_choice_skips_hypotheses_with_annotator_disagreement():
    rows = [
        {"id": "e", "premise": "p", "hypothesis": "h_e", "label": "entailment", "group": "g", "label_source": "human_multi",
         "target": {"entailment": 0.6, "neutral": 0.4}},
        {"id": "e2", "premise": "p", "hypothesis": "h_e2", "label": "entailment", "group": "g", "label_source": "human_multi",
         "target": {"entailment": 1.0}},
        {"id": "c", "premise": "p", "hypothesis": "h_c", "label": "contradiction", "group": "g", "label_source": "human"},
    ]
    items = nli_items(rows, "t", "MIT", "train", Report("t"))
    prem = [it for it in items if it.meta["format"] == "nli_premise_choice"]
    assert [it.id for it in prem] == ["t:premise:e2"]
    assert "h_e" not in prem[0].criteria.values()
