"""Release tooling for nGibbs; the steps of .github/workflows/release.yml.

    python scripts/release.py plan 1.1.0 [--json plan.json] [--summary FILE]
    python scripts/release.py stage plan.json [--github-output FILE]
    python scripts/release.py smoke plan.json          # run with an installed wheel
    python scripts/release.py merge PR_NUMBER --version 1.1.0
    python scripts/release.py close PR_NUMBER [--reason TEXT]

plan   Checks the version (well formed, not on PyPI or tagged yet, newer than
       PyPI's latest) and compares every model working copy in
       src/ngibbs/engine/TrainedModels/ with Hugging Face. New or changed
       models are only allowed in X.Y.0 releases (and their pre-releases), and
       need a model card. Prints what will be uploaded and fails on any problem.
       Run it locally before releasing; it changes nothing.
stage  Opens one Hugging Face pull request with the new model folders and the
       updated index.json. Nothing reaches the repo's main branch yet.
smoke  Loads every model through the installed wheel, from the staging pull
       request (NGIBBS_HF_REVISION=refs/pr/N), and checks the version it gets.
merge  Merges the staging pull request: the models go live. Run after PyPI.
close  Closes the staging pull request: dry runs and failed releases.

Model versions on Hugging Face are never changed once final. A pre-release
(X.Y.0aN/bN/rcN) may replace its own vX.Y until X.Y.0 itself is released.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PR_PREFIX = "nGibbs release "
RELEASE_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)((a|b|rc)\d+)?$")
UPLOAD_KINDS = ("new", "changed", "replace", "carry")
HF_KINDS = UPLOAD_KINDS + ("finalize", "readme")


def _registry():
    """ngibbs.registry from this checkout (plan/stage/merge/close need no
    install, and no torch)."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from ngibbs import registry
    return registry


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def pypi_versions() -> list:
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/nGibbs/json", timeout=20) as r:
            return sorted(json.loads(r.read())["releases"])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise


def git_tag_exists(tag: str) -> bool:
    for cmd in (["git", "-C", str(ROOT), "ls-remote", "--tags", "origin", f"refs/tags/{tag}"],
                ["git", "-C", str(ROOT), "tag", "--list", tag]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout
            return bool(out.strip())
        except Exception:
            continue
    return False


def make_plan(version: str, index: dict, local: dict, pypi: list, tag_exists: bool) -> dict:
    """The release plan. ``local``: {model: {'fingerprint', 'card_sha256',
    'card_problems'}} for each working copy. Pure, so it can be unit tested."""
    reg = _registry()
    problems, warnings_, actions = [], [], []
    m = RELEASE_RE.match(version)
    if not m:
        return {"version": version, "ok": False, "actions": [], "warnings": [],
                "problems": [f"'{version}' is not a release version: use X.Y.Z or "
                             "X.Y.Z{a,b,rc}N, e.g. 1.1.0 or 1.1.0rc1"]}
    X, Y, Z, pre = int(m.group(1)), int(m.group(2)), int(m.group(3)), bool(m.group(4))
    target = f"{X}.{Y}"

    from packaging.version import Version
    if version in pypi:
        problems.append(f"nGibbs {version} is already on PyPI; a version can only be used once.")
    newer = [p for p in pypi if Version(p) >= Version(version)]
    if newer and version not in pypi:
        problems.append(f"PyPI already has {max(newer, key=Version)}, which is not older than {version}.")
    if tag_exists:
        problems.append(f"Git tag v{version} already exists.")

    hub = index.get("models", {})
    for model in sorted(local):
        info = local[model]
        versions = hub.get(model, {})
        latest = max(versions, key=reg.version_key) if versions else None
        le = versions.get(latest, {}) if latest else {}
        a = {"model": model, "target": target, "latest": latest}
        differs = latest is None or le.get("fingerprint") != info["fingerprint"]
        if differs and Z != 0:
            problems.append(f"{model}: model files are {'new' if latest is None else 'changed'}, but "
                            f"{version} is a patch release. Models ship only in X.Y.0 releases "
                            f"(here {X}.{Y + 1}.0, or {X + 1}.0.0).")
            kind = "blocked"
        elif latest is None:
            kind = "new"
        elif differs:
            if reg.version_key(latest) > reg.version_key(target):
                problems.append(f"{model}: Hugging Face already has {model}v{latest}, "
                                f"newer than this release ({version}).")
                kind = "unchanged"
            elif latest == target and not le.get("prerelease"):
                problems.append(f"{model}: files changed, but {model}v{target} is already final; "
                                "new models need a new minor release.")
                kind = "unchanged"
            else:
                kind = "replace" if latest == target else "changed"
        elif reg.version_key(latest)[0] < X:
            kind = "carry"
        elif latest == target and le.get("prerelease") and not pre:
            kind = "finalize"
            a["readme_changed"] = le.get("card_sha256") != info["card_sha256"]
        elif le.get("card_sha256") != info["card_sha256"]:
            kind = "readme"
        else:
            kind = "unchanged"
        a["kind"] = kind
        if kind in UPLOAD_KINDS and Z != 0:      # carry-forward in a patch release
            problems.append(f"{model}: {version} is a patch release; models ship only in X.Y.0 releases.")
        if kind in HF_KINDS:
            problems += info["card_problems"]
        actions.append(a)

    for model in sorted(set(hub) - set(local)):
        latest = max(hub[model], key=reg.version_key)
        actions.append({"model": model, "kind": "absent", "latest": latest, "target": None})
        if reg.version_key(latest)[0] < X:
            warnings_.append(f"{model}v{latest} is not in this checkout, and nGibbs {X}.x will not "
                             "load it (a major release republishes only the models it has).")

    expected = {}
    for a in actions:
        if a["kind"] in ("absent", "blocked"):
            continue
        expected[a["model"]] = a["target"] if a["kind"] in UPLOAD_KINDS + ("finalize",) else a["latest"]
    return {"version": version, "prerelease": pre, "target": target, "actions": actions,
            "expected": expected, "problems": problems, "warnings": warnings_, "ok": not problems}


def plan_lines(plan: dict) -> list:
    """Plain-text description of each model action."""
    out = []
    v = plan["version"]
    for a in plan["actions"]:
        model, k, t, last = a["model"], a["kind"], a.get("target"), a.get("latest")
        pre = " (pre-release)" if plan.get("prerelease") else ""
        if k == "new":
            out.append(f"New model detected: {model} -> will be uploaded to Hugging Face as {model}v{t}{pre}")
        elif k == "changed":
            out.append(f"Changed model detected: {model}v{last} -> {model}v{t}: will be uploaded to "
                       f"Hugging Face as {model}v{t}{pre}; make_ngibbs('{model}') gives {model}v{t} "
                       f"from nGibbs {t} on")
        elif k == "replace":
            out.append(f"Changed pre-release model: {model}v{t} will be replaced{pre}")
        elif k == "carry":
            out.append(f"Carried forward: {model}v{last} -> {model}v{t} (same files, republished "
                       f"for nGibbs {t.split('.')[0]}.x)")
        elif k == "finalize":
            out.append(f"Finalized: pre-release {model}v{t} becomes final with nGibbs {v}"
                       + (" (model card updated)" if a.get("readme_changed") else ""))
        elif k == "readme":
            out.append(f"Model card updated: {model}v{last} (README.md only)")
        elif k == "unchanged":
            out.append(f"Unchanged: {model} (stays {model}v{last})")
        elif k == "blocked":
            out.append(f"Blocked: {model} has new or changed files, which a patch release cannot ship")
        elif k == "absent":
            out.append(f"Not in this checkout, left as is on Hugging Face: {model} (latest {model}v{last})")
    return out


def plan_markdown(plan: dict) -> str:
    md = [f"## nGibbs {plan['version']}: release plan", ""]
    if plan["problems"]:
        md += ["### Problems (the release stops here)", ""] + [f"- {p}" for p in plan["problems"]] + [""]
    md += ["### Models", ""] + [f"- {line}" for line in plan_lines(plan)] + [""]
    if plan["warnings"]:
        md += ["### Warnings", ""] + [f"- {w}" for w in plan["warnings"]] + [""]
    return "\n".join(md)


def local_state(reg) -> dict:
    out = {}
    for model, d in reg.checkout_models().items():
        man = reg.bundle_manifest(d)
        out[model] = {"fingerprint": man["fingerprint"],
                      "card_sha256": man["files"].get(reg.CARD_FILE, {}).get("sha256"),
                      "card_problems": reg.card_problems(d)}
    return out


def _index_sha(index: dict) -> str:
    return hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()


def cmd_plan(args) -> int:
    reg = _registry()
    version = args.version.removeprefix("v")
    index = reg.fetch_index(strict=True, revision="main")
    plan = make_plan(version, index, local_state(reg), pypi_versions(), git_tag_exists(f"v{version}"))
    plan["index_sha256"] = _index_sha(index)
    md = plan_markdown(plan)
    plan["summary_md"] = md
    print(md)
    if args.json:
        Path(args.json).write_text(json.dumps(plan, indent=2))
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(md + "\n")
    return 0 if plan["ok"] else 1


# --------------------------------------------------------------------------- #
# stage / merge / close
# --------------------------------------------------------------------------- #
def _hf():
    from huggingface_hub import HfApi
    return HfApi()          # token from $HF_TOKEN


def _remote_json(reg, path: str, revision: str = "main") -> dict:
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(reg.HF_REPO_ID, path, revision=revision)).read_text())


def _write_output(path, **kv):
    if path:
        with open(path, "a") as f:
            for k, v in kv.items():
                f.write(f"{k}={v}\n")


def cmd_stage(args) -> int:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    reg = _registry()
    plan = json.loads(Path(args.plan).read_text())
    if not plan["ok"]:
        print("The plan has problems; not staging.", file=sys.stderr)
        return 1
    acts = [a for a in plan["actions"] if a["kind"] in HF_KINDS]
    if not acts:
        print("No model changes: nothing to stage on Hugging Face.")
        _write_output(args.github_output, pr="")
        return 0

    api, repo, version = _hf(), reg.HF_REPO_ID, plan["version"]
    index = reg.fetch_index(strict=True, revision="main")
    if _index_sha(index) != plan["index_sha256"]:
        print("Hugging Face's index.json changed since the plan was made; plan again.", file=sys.stderr)
        return 1
    for d in api.get_repo_discussions(repo, discussion_type="pull_request", discussion_status="open"):
        if d.title.startswith(PR_PREFIX):
            api.change_discussion_status(repo, d.num, "closed",
                                         comment="Superseded by a newer release attempt.")
            print(f"Closed stale staging pull request #{d.num} ({d.title}).")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    local = reg.checkout_models()
    ops = []
    for a in acts:
        model, kind = a["model"], a["kind"]
        d = local[model]
        man = reg.bundle_manifest(d)
        card = reg.read_card(d)
        card_sha = man["files"][reg.CARD_FILE]["sha256"]
        entries = index["models"].setdefault(model, {})
        if kind in UPLOAD_KINDS:
            prefix = f"{model}/v{a['target']}"
            for rel in man["files"]:
                ops.append(CommitOperationAdd(f"{prefix}/{rel}", str(d / rel)))
            if kind == "replace":
                old = _remote_json(reg, f"{prefix}/{reg.MANIFEST_FILE}")
                for rel in set(old.get("files", {})) - set(man["files"]):
                    ops.append(CommitOperationDelete(f"{prefix}/{rel}"))
            files = man["files"]
            entry = {
                "released_with": version, "prerelease": plan["prerelease"],
                "api": card.get("api"), "summary": card.get("summary"),
                "fingerprint": man["fingerprint"], "card_sha256": card_sha,
                "size_bytes": sum(f["size"] for f in files.values()),
                "download_bytes": sum(f["size"] for r, f in files.items() if not reg.is_on_demand(r)),
                "n_files": len(files) + 1, "uploaded_utc": now,
            }
            if card.get("requires_ngibbs"):
                entry["requires_ngibbs"] = card["requires_ngibbs"]
            entries[a["target"]] = entry
            manifest = {"model": model, "version": a["target"], "released_with": version,
                        "fingerprint": man["fingerprint"], "files": files, "index_entry": entry}
            ops.append(CommitOperationAdd(f"{prefix}/{reg.MANIFEST_FILE}",
                                          json.dumps(manifest, indent=1).encode()))
        else:
            ver = a["target"] if kind == "finalize" else a["latest"]
            entry = entries[ver]
            if kind == "finalize":
                entry["prerelease"] = False
                entry["released_with"] = version
            if kind == "readme" or a.get("readme_changed"):
                prefix = f"{model}/v{ver}"
                ops.append(CommitOperationAdd(f"{prefix}/{reg.CARD_FILE}", str(d / reg.CARD_FILE)))
                entry["card_sha256"] = card_sha
                entry["summary"] = card.get("summary", entry.get("summary"))
                old = _remote_json(reg, f"{prefix}/{reg.MANIFEST_FILE}")
                old["files"][reg.CARD_FILE] = man["files"][reg.CARD_FILE]
                old["index_entry"] = entry
                ops.append(CommitOperationAdd(f"{prefix}/{reg.MANIFEST_FILE}",
                                              json.dumps(old, indent=1).encode()))
    index["format"] = reg.INDEX_FORMAT
    index["updated_utc"] = now
    ops.append(CommitOperationAdd(reg.INDEX_FILE, json.dumps(index, indent=1, sort_keys=True).encode()))

    info = api.create_commit(repo, operations=ops, commit_message=f"{PR_PREFIX}{version}",
                             commit_description=plan.get("summary_md", ""), create_pr=True)
    print(f"Staged on Hugging Face as pull request #{info.pr_num}: {info.pr_url}")
    _write_output(args.github_output, pr=info.pr_num)
    return 0


def cmd_merge(args) -> int:
    reg = _registry()
    _hf().merge_pull_request(reg.HF_REPO_ID, int(args.pr),
                             comment=f"Released with nGibbs {args.version}.")
    print(f"Merged Hugging Face pull request #{args.pr}: the models are live.")
    return 0


def cmd_close(args) -> int:
    reg = _registry()
    _hf().change_discussion_status(reg.HF_REPO_ID, int(args.pr), "closed", comment=args.reason)
    print(f"Closed Hugging Face pull request #{args.pr}.")
    return 0


# --------------------------------------------------------------------------- #
# smoke (installed wheel)
# --------------------------------------------------------------------------- #
def cmd_smoke(args) -> int:
    import ngibbs
    from ngibbs import registry as reg
    plan = json.loads(Path(args.plan).read_text())
    print(f"ngibbs {reg.installed_version()} from {Path(ngibbs.__file__).parent}; "
          f"Hugging Face revision {reg.hub_revision()}")
    failures = []
    if reg.installed_version() != plan["version"]:
        failures.append(f"installed version {reg.installed_version()} != {plan['version']}")
    if reg.checkout_models():
        failures.append("the wheel contains model working copies; it must not")
    for model, expected in sorted(plan["expected"].items()):
        try:
            api = ngibbs.make_ngibbs(model)
            subs = [api.nocr, api.cr] if hasattr(api, "nocr") else [api]
            if api.model_version != expected:
                failures.append(f"{model}: loaded v{api.model_version}, expected v{expected}")
            if api.standard_workspace_dir is None:
                failures.append(f"{model}: no standards/ folder")
            if not all(s._test_bundles for s in subs):
                failures.append(f"{model}: missing validation datasets")
            print(f"  {model}: v{api.model_version} OK")
        except Exception as e:
            failures.append(f"{model}: {type(e).__name__}: {e}")
    ngibbs.available_models("all")
    for f in failures:
        print(f"FAILED: {f}", file=sys.stderr)
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("version")
    p.add_argument("--json")
    p.add_argument("--summary", help="append the Markdown summary to this file")
    p = sub.add_parser("stage")
    p.add_argument("plan")
    p.add_argument("--github-output")
    p = sub.add_parser("smoke")
    p.add_argument("plan")
    p = sub.add_parser("merge")
    p.add_argument("pr")
    p.add_argument("--version", required=True)
    p = sub.add_parser("close")
    p.add_argument("pr")
    p.add_argument("--reason", default="Release did not complete; staged models discarded.")
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "stage": cmd_stage, "smoke": cmd_smoke,
            "merge": cmd_merge, "close": cmd_close}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
