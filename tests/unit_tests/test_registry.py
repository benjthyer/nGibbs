"""Model registry rules (names, compatibility, version choice, model cards) and
the release plan (scripts/release.py). No network, no model weights."""
import importlib.util
import warnings
from pathlib import Path

import pytest

from ngibbs import registry as reg

REPO = Path(__file__).resolve().parents[2]


def _release_module():
    spec = importlib.util.spec_from_file_location("ngibbs_release", REPO / "scripts" / "release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# names and compatibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec, expected", [
    ("MELTS120", ("MELTS120", None)),
    ("MELTS120v1.2", ("MELTS120", "1.2")),
    ("HeFESToEarthv10.11", ("HeFESToEarth", "10.11")),
    ("HeFESToMars", ("HeFESToMars", None)),
])
def test_parse_name(spec, expected):
    assert reg.parse_name(spec) == expected


@pytest.mark.parametrize("model_version, installed, reason", [
    ("1.2", "1.2.0", None),
    ("1.2", "1.2.3", None),
    ("1.0", "1.4.1", None),                          # older minor: fine
    ("1.3", "1.2.2", "needs ngibbs>=1.3"),           # newer than the install
    ("1.3", "2.1.0", "needs ngibbs<2"),              # other major series
    ("1.2", "1.2.0rc1", None),                       # pre-releases count as X.Y
    ("1.2", "1.2.1.dev3+g1234", None),               # development builds too
])
def test_incompatibility_default_rule(model_version, installed, reason):
    assert reg.incompatibility(model_version, {}, installed) == reason


def test_incompatibility_card_override():
    entry = {"requires_ngibbs": ">=1.3,<3"}
    assert reg.incompatibility("1.3", entry, "2.1.0") is None
    assert reg.incompatibility("1.3", entry, "1.2.0") == "needs ngibbs>=1.3,<3"


def test_model_version_for():
    assert reg.model_version_for("1.2.0") == "1.2"
    assert reg.model_version_for("1.2.0rc1") == "1.2"


INDEX = {"models": {"MELTS120": {"1.0": {}, "1.1": {}, "1.3": {}}}}


def test_resolve_newest_compatible_warns_about_newer():
    with pytest.warns(UserWarning, match=r"MELTS120v1.3 is available but needs ngibbs>=1.3"):
        v, _ = reg.resolve("MELTS120", None, INDEX, "1.2.0")
    assert v == "1.1"


def test_resolve_explicit_and_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert reg.resolve("MELTS120", "1.0", INDEX, "1.2.0")[0] == "1.0"
        assert reg.resolve("MELTS120", None, INDEX, "1.3.0")[0] == "1.3"
    with pytest.raises(RuntimeError, match="pip install --upgrade"):
        reg.resolve("MELTS120", "1.3", INDEX, "1.2.0")
    with pytest.raises(KeyError, match="no version v1.2"):
        reg.resolve("MELTS120", "1.2", INDEX, "1.2.0")
    with pytest.raises(KeyError, match="No model named"):
        reg.resolve("MELTS999", None, INDEX, "1.2.0")
    with pytest.raises(RuntimeError, match="No version of MELTS120 works"):
        reg.resolve("MELTS120", None, INDEX, "0.9.0")


# --------------------------------------------------------------------------- #
# model cards and bundles
# --------------------------------------------------------------------------- #
def _bundle(tmp_path, name="MELTS120", card="---\nmodel: MELTS120\napi: MELTSAPI\nsummary: s\n---\nText\n"):
    d = tmp_path / name
    d.mkdir()
    (d / "a.tar").write_bytes(b"weights")
    if card is not None:
        (d / "README.md").write_text(card)
    return d


def test_card_ok_and_problems(tmp_path):
    assert reg.card_problems(_bundle(tmp_path)) == []
    assert reg.read_card(tmp_path / "MELTS120")["api"] == "MELTSAPI"
    assert "no README.md" in reg.card_problems(_bundle(tmp_path, "A", card=None))[0]
    assert "no YAML header" in reg.card_problems(_bundle(tmp_path, "B", card="# Just text\n"))[0]
    probs = reg.card_problems(_bundle(tmp_path, "C", card="---\nmodel: X\napi: FooAPI\n---\n"))
    assert any("lacks 'summary'" in p for p in probs)
    assert any("folder is 'C'" in p for p in probs)
    assert any("FooAPI" in p for p in probs)


def test_fingerprint_ignores_model_card(tmp_path):
    d = _bundle(tmp_path)
    (d / "Thumbs:Zone.Identifier").write_text("junk")          # never part of a bundle
    fp = reg.bundle_manifest(d)["fingerprint"]
    assert set(reg.bundle_manifest(d)["files"]) == {"README.md", "a.tar"}
    (d / "README.md").write_text("---\nmodel: MELTS120\napi: MELTSAPI\nsummary: new text\n---\n")
    assert reg.bundle_manifest(d)["fingerprint"] == fp
    (d / "a.tar").write_bytes(b"new weights")
    assert reg.bundle_manifest(d)["fingerprint"] != fp


def test_checkout_model_cards_are_valid():
    models = reg.checkout_models()
    if not models:
        pytest.skip("not a source checkout")
    for d in models.values():
        assert reg.card_problems(d) == []


# --------------------------------------------------------------------------- #
# release plan
# --------------------------------------------------------------------------- #
def _local(fp="F1", card="C1", problems=()):
    return {"fingerprint": fp, "card_sha256": card, "card_problems": list(problems)}


def _entry(fp="F1", card="C1", pre=False):
    return {"fingerprint": fp, "card_sha256": card, "prerelease": pre}


def _plan(version, hub, local, pypi=("0.1.0b4",), tag=False):
    rel = _release_module()
    plan = rel.make_plan(version, {"models": hub}, local, list(pypi), tag)
    return plan, {a["model"]: a for a in plan["actions"]}, rel.plan_lines(plan)


def test_plan_inaugural_release_uploads_everything_as_v1_0():
    plan, acts, lines = _plan("1.0.0", {}, {"MELTS120": _local(), "HeFESToEarth": _local()})
    assert plan["ok"], plan["problems"]
    assert {a["kind"] for a in acts.values()} == {"new"}
    assert plan["expected"] == {"MELTS120": "1.0", "HeFESToEarth": "1.0"}
    assert "New model detected: MELTS120 -> will be uploaded to Hugging Face as MELTS120v1.0" in lines


def test_plan_changed_model_in_minor_release():
    plan, acts, lines = _plan("1.1.0", {"MELTS120": {"1.0": _entry()}, "MELTS102": {"1.0": _entry()}},
                              {"MELTS120": _local(fp="F2"), "MELTS102": _local()}, pypi=["1.0.0"])
    assert plan["ok"]
    assert acts["MELTS120"]["kind"] == "changed" and acts["MELTS102"]["kind"] == "unchanged"
    assert plan["expected"] == {"MELTS120": "1.1", "MELTS102": "1.0"}
    assert any(l.startswith("Changed model detected: MELTS120v1.0 -> MELTS120v1.1") for l in lines)


def test_plan_patch_release_rejects_model_changes():
    plan, acts, _ = _plan("1.1.1", {"MELTS120": {"1.1": _entry()}}, {"MELTS120": _local(fp="F2")},
                          pypi=["1.1.0"])
    assert not plan["ok"]
    assert any("patch release" in p for p in plan["problems"])


def test_plan_patch_release_allows_model_card_updates():
    plan, acts, _ = _plan("1.1.1", {"MELTS120": {"1.1": _entry()}}, {"MELTS120": _local(card="C2")},
                          pypi=["1.1.0"])
    assert plan["ok"] and acts["MELTS120"]["kind"] == "readme"


def test_plan_requires_model_card_for_new_models():
    plan, _, _ = _plan("1.0.0", {}, {"MELTS120": _local(problems=["MELTS120: no README.md (model card)"])})
    assert not plan["ok"]
    assert "MELTS120: no README.md (model card)" in plan["problems"]


def test_plan_prerelease_can_be_replaced_then_finalized():
    hub = {"MELTS120": {"1.0": _entry(), "1.1": _entry(fp="F2", pre=True)}}
    plan, acts, _ = _plan("1.1.0rc2", hub, {"MELTS120": _local(fp="F3")}, pypi=["1.0.0", "1.1.0rc1"])
    assert plan["ok"] and acts["MELTS120"]["kind"] == "replace"
    plan, acts, _ = _plan("1.1.0", hub, {"MELTS120": _local(fp="F2")}, pypi=["1.0.0", "1.1.0rc1"])
    assert plan["ok"] and acts["MELTS120"]["kind"] == "finalize"
    final = {"MELTS120": {"1.1": _entry(fp="F2", pre=False)}}
    plan, _, _ = _plan("1.1.0", final, {"MELTS120": _local(fp="F3")}, pypi=["1.0.0"])
    assert not plan["ok"] and any("already final" in p for p in plan["problems"])


def test_plan_major_release_carries_models_forward_and_warns_about_absent_ones():
    hub = {"MELTS120": {"1.3": _entry()}, "OldModel": {"1.0": _entry()}}
    plan, acts, _ = _plan("2.0.0", hub, {"MELTS120": _local()}, pypi=["1.3.0"])
    assert plan["ok"]
    assert acts["MELTS120"]["kind"] == "carry" and plan["expected"]["MELTS120"] == "2.0"
    assert acts["OldModel"]["kind"] == "absent"
    assert any("OldModel" in w for w in plan["warnings"])


def test_plan_version_checks():
    assert not _plan("1.0", {}, {})[0]["ok"]                              # malformed
    assert not _plan("1.0.0", {}, {}, pypi=["1.0.0"])[0]["ok"]            # on PyPI already
    assert not _plan("1.0.0", {}, {}, pypi=["1.1.0"])[0]["ok"]            # older than PyPI's latest
    assert not _plan("1.0.0", {}, {}, tag=True)[0]["ok"]                  # tagged already
    assert _plan("1.0.0", {}, {})[0]["ok"]
