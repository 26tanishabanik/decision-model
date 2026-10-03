"""Checks on the built training-source files (run scripts/build_train_data.py first)."""
import json
import random
from collections import defaultdict

import pytest

from decision_model.schema import DATA_DIR, LABEL_SOURCES, ROOT, SOURCES_DIR, Item, validate

MANIFEST = SOURCES_DIR / "manifest.json"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="training sources not built (scripts/build_train_data.py)")


def _manifest():
    return json.loads(MANIFEST.read_text())


def _files():
    out = []
    if not MANIFEST.exists():
        return out
    for name, e in _manifest()["sources"].items():
        for kind in ("pool", "native_test"):
            out.append((name, kind, ROOT / e["files"][kind]["path"]))
    return out


@pytest.fixture(scope="module")
def index():
    """(source, kind) -> list of (id, group, split); one streaming pass over every built file."""
    idx = {}
    for name, kind, path in _files():
        rows = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                rows.append((d["id"], d["group"], d["split"], d["source"]))
        idx[(name, kind)] = rows
    return idx


@pytest.mark.parametrize("name,kind,path", _files(), ids=lambda x: str(x) if not hasattr(x, "name") else "")
def test_sample_schema_valid(name, kind, path):
    with open(path) as f:
        lines = f.readlines()
    rng = random.Random(0)
    sample = lines[:50] + rng.sample(lines, min(500, len(lines)))
    for line in sample:
        it = Item(**json.loads(line))
        validate(it)
        assert it.source == name
        assert it.state.strip()
        assert it.label_source in LABEL_SOURCES
        if kind == "pool":
            assert it.split in ("train", "dev", "calib")
        else:
            assert it.split == "eval"
        if it.meta.get("n_options") is not None:
            assert it.meta["n_options"] == len(it.criteria)


def test_manifest_matches_files():
    import hashlib
    for name, e in _manifest()["sources"].items():
        for kind in ("pool", "native_test"):
            fe = e["files"][kind]
            data = (ROOT / fe["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == fe["sha256"], (name, kind)
            assert data.count(b"\n") == fe["n"], (name, kind)
        assert e["revision"], name
        assert e["group_definition"], name


def test_groups_never_straddle_splits(index):
    for (name, kind), rows in index.items():
        if kind != "pool":
            continue
        split_of_group = {}
        for _, g, s, _ in rows:
            assert split_of_group.setdefault(g, s) == s, (name, g)
        assert {"train", "dev", "calib"} <= set(split_of_group.values()), name


def test_no_id_collisions_across_sources(index):
    seen = {}
    for (name, kind), rows in index.items():
        for i, _, _, src in rows:
            assert src == name
            assert i.startswith(f"{name}:"), i
            assert i not in seen, (i, seen.get(i), (name, kind))
            seen[i] = (name, kind)


def test_native_test_not_in_pool(index):
    by_source = defaultdict(dict)
    for (name, kind), rows in index.items():
        by_source[name][kind] = {i for i, *_ in rows}
    for name, d in by_source.items():
        assert not (d["pool"] & d["native_test"]), name


def test_item_licenses_match_register():
    reg = {e["dataset"]: e for e in json.loads((DATA_DIR / "licenses.json").read_text())["sources"]}
    for name, e in _manifest()["sources"].items():
        ids = reg[e["license"]["dataset"]]["licenses"]
        for lic in e["item_licenses"]:
            assert lic in ids or any(x.startswith(lic + " ") for x in ids), (name, lic)
