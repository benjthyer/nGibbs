"""Trained-model registry: model names, versions, compatibility and downloads.

Every released model lives in the Hugging Face model repo ``HF_REPO_ID``, one
folder per version, and is never overwritten once its release is final::

    index.json                       every model and version (read by available_models)
    <Model>/v<X.Y>/README.md         model card; YAML header: model, api, summary, ...
    <Model>/v<X.Y>/manifest.json     SHA-256 and size of every file in the bundle
    <Model>/v<X.Y>/*.tar, *.pt       checkpoints and temperature network
    <Model>/v<X.Y>/*_Test_subset*.tar.gz   validation datasets (api.test())
    <Model>/v<X.Y>/standards/        standard simulator runs (api.evaluate_workspace())

A model version is named after the nGibbs minor release that published it
(``MELTS120v1.2`` was released with nGibbs 1.2.0). An installed nGibbs accepts
a model version from its own major series that is not newer than itself: nGibbs
1.2.x loads v1.0-v1.2, not v1.3 or v2.0. A model card can override this with a
``requires_ngibbs`` specifier.

Downloads go to ``~/.cache/ngibbs/models/<Model>/v<X.Y>/`` (``$NGIBBS_MODELS_DIR``
moves it). In a source checkout, each model's working copy in
``src/ngibbs/engine/TrainedModels/<Model>/`` is used instead when no version is
requested; releases publish those working copies (scripts/release.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HF_REPO_ID = "benjthyer/nGibbs"
INDEX_FILE = "index.json"
CARD_FILE = "README.md"
MANIFEST_FILE = "manifest.json"
INDEX_FORMAT = 1

# Bundle subfolders fetched only when asked for (not by make_ngibbs).
ON_DEMAND_DIRS = ("training_sample",)
KNOWN_APIS = ("MELTSAPI", "HeFESToAPI")
REQUIRED_CARD_FIELDS = ("model", "api", "summary")

_PKG_DIR = Path(__file__).resolve().parent
CHECKOUT_MODELS_DIR = _PKG_DIR / "engine" / "TrainedModels"
REPO_ROOT = _PKG_DIR.parent.parent           # meaningful in a source checkout only
BASELINES_DIR = REPO_ROOT / "benchmarks"

# Files that are never part of a bundle (Windows/macOS sidecars, local outputs).
_JUNK = re.compile(r"(:Zone\.Identifier|\.Identifier|\.pyc|\.DS_Store)$"
                   r"|(^|/)(__pycache__|plots|deployment_test)(/|$)")
_NAME_RE = re.compile(r"^(?P<model>.+?)(?:v(?P<version>\d+\.\d+))?$")
_CARD_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(\r?\n|\Z)", re.DOTALL)


# --------------------------------------------------------------------------- #
# names and versions
# --------------------------------------------------------------------------- #
def parse_name(spec: str) -> Tuple[str, Optional[str]]:
    """``'MELTS120v1.2'`` -> ``('MELTS120', '1.2')``; ``'MELTS120'`` -> ``('MELTS120', None)``."""
    m = _NAME_RE.match(spec.strip())
    if not m:
        raise ValueError(f"Not a model name: {spec!r}")
    return m.group("model"), m.group("version")


def version_key(version: str) -> Tuple[int, int]:
    """``'1.12'`` -> ``(1, 12)``, for sorting model versions."""
    major, minor = version.split(".")
    return int(major), int(minor)


def model_version_for(release: str) -> str:
    """The model version an nGibbs release publishes: ``'1.2.0rc1'`` -> ``'1.2'``."""
    from packaging.version import Version
    r = Version(release).release
    return f"{r[0]}.{r[1] if len(r) > 1 else 0}"


def installed_version() -> Optional[str]:
    """This nGibbs's version, or None if it cannot be determined."""
    try:
        from ._version import __version__
        return __version__
    except ImportError:
        pass
    try:
        from importlib.metadata import version
        return version("nGibbs")
    except Exception:
        return None


def incompatibility(model_version: str, entry: dict, installed: Optional[str]) -> Optional[str]:
    """None if nGibbs ``installed`` can load ``model_version``, else the reason,
    e.g. ``'needs ngibbs>=1.3'``."""
    if installed is None:
        return None
    from packaging.version import Version
    iv = Version(installed)
    spec = entry.get("requires_ngibbs")
    if spec:
        from packaging.specifiers import SpecifierSet
        return None if SpecifierSet(spec).contains(iv.base_version) else f"needs ngibbs{spec}"
    mj, mn = version_key(model_version)
    r = iv.release
    ij, imn = r[0], (r[1] if len(r) > 1 else 0)
    if ij > mj:
        return f"needs ngibbs<{mj + 1}"
    if (ij, imn) < (mj, mn):
        return f"needs ngibbs>={mj}.{mn}"
    return None


# --------------------------------------------------------------------------- #
# model cards and bundle contents
# --------------------------------------------------------------------------- #
def read_card(bundle_dir: Path) -> dict:
    """The YAML header of ``bundle_dir/README.md`` ({} if there is none)."""
    path = Path(bundle_dir) / CARD_FILE
    if not path.is_file():
        return {}
    m = _CARD_RE.match(path.read_text(encoding="utf-8"))
    if not m:
        return {}
    import yaml
    meta = yaml.safe_load(m.group(1)) or {}
    return meta if isinstance(meta, dict) else {}


def card_problems(bundle_dir: Path) -> List[str]:
    """What is wrong with a bundle's model card (empty if nothing)."""
    bundle_dir = Path(bundle_dir)
    if not (bundle_dir / CARD_FILE).is_file():
        return [f"{bundle_dir.name}: no {CARD_FILE} (model card)"]
    try:
        meta = read_card(bundle_dir)
    except Exception as e:
        return [f"{bundle_dir.name}: {CARD_FILE} YAML header does not parse: {e}"]
    if not meta:
        return [f"{bundle_dir.name}: {CARD_FILE} has no YAML header (--- ... ---)"]
    out = [f"{bundle_dir.name}: {CARD_FILE} header lacks '{k}'"
           for k in REQUIRED_CARD_FIELDS if not meta.get(k)]
    if meta.get("model") and meta["model"] != bundle_dir.name:
        out.append(f"{bundle_dir.name}: {CARD_FILE} says model '{meta['model']}', "
                   f"but the folder is '{bundle_dir.name}'")
    if meta.get("api") and meta["api"] not in KNOWN_APIS:
        out.append(f"{bundle_dir.name}: api '{meta['api']}' is not one of {KNOWN_APIS}")
    if meta.get("requires_ngibbs"):
        try:
            from packaging.specifiers import SpecifierSet
            SpecifierSet(meta["requires_ngibbs"])
        except Exception:
            out.append(f"{bundle_dir.name}: requires_ngibbs '{meta['requires_ngibbs']}' "
                       "is not a version specifier")
    return out


def bundle_files(bundle_dir: Path) -> List[str]:
    """A bundle's files, relative to it, '/'-separated. In a git checkout only
    files git would commit count (tracked, or new and not ignored), so local
    outputs never get published."""
    bundle_dir = Path(bundle_dir)
    try:
        out = subprocess.run(["git", "-C", str(bundle_dir), "ls-files", "-z", "--cached",
                              "--others", "--exclude-standard", "--", "."],
                             capture_output=True, check=True, timeout=60).stdout
        files = [f for f in out.decode().split("\0") if f]
        if files:
            return sorted(f for f in files if (bundle_dir / f).is_file() and not _JUNK.search(f))
    except Exception:
        pass
    return sorted(p.relative_to(bundle_dir).as_posix() for p in bundle_dir.rglob("*")
                  if p.is_file() and not _JUNK.search(p.relative_to(bundle_dir).as_posix()))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(files: Dict[str, dict]) -> str:
    """One hash for a bundle's contents. The model card and manifest are left
    out, so editing a README does not make a new model version."""
    h = hashlib.sha256()
    for rel in sorted(files):
        if rel in (CARD_FILE, MANIFEST_FILE):
            continue
        h.update(f"{rel}\t{files[rel]['sha256']}\n".encode())
    return h.hexdigest()


_MANIFEST_CACHE: Dict[tuple, dict] = {}


def bundle_manifest(bundle_dir: Path) -> dict:
    """``{'files': {rel: {'sha256', 'size'}}, 'fingerprint': ...}`` for a local bundle."""
    bundle_dir = Path(bundle_dir).resolve()
    rels = [r for r in bundle_files(bundle_dir) if r != MANIFEST_FILE]
    stamp = tuple((r, (bundle_dir / r).stat().st_size, (bundle_dir / r).stat().st_mtime_ns)
                  for r in rels)
    key = (str(bundle_dir), stamp)
    if key not in _MANIFEST_CACHE:
        files = {r: {"sha256": file_sha256(bundle_dir / r), "size": (bundle_dir / r).stat().st_size}
                 for r in rels}
        _MANIFEST_CACHE[key] = {"files": files, "fingerprint": fingerprint(files)}
    return _MANIFEST_CACHE[key]


def is_on_demand(rel: str) -> bool:
    return rel.split("/", 1)[0] in ON_DEMAND_DIRS


# --------------------------------------------------------------------------- #
# source checkout
# --------------------------------------------------------------------------- #
def _is_bundle(d: Path) -> bool:
    return d.is_dir() and ((d / CARD_FILE).is_file() or any(d.glob("*.tar")))


def checkout_models() -> Dict[str, Path]:
    """{model name: working-copy folder} in a source checkout ({} when installed)."""
    if not CHECKOUT_MODELS_DIR.is_dir():
        return {}
    return {d.name: d for d in sorted(CHECKOUT_MODELS_DIR.iterdir()) if _is_bundle(d)}


def baseline_dir(model: str, device: str = "cpu") -> Path:
    """``benchmarks/<model>/<cpu|cuda>/`` in a source checkout: the regression
    baseline for the model's working copy."""
    return BASELINES_DIR / model / ("cuda" if str(device).startswith("cuda") else "cpu")


# --------------------------------------------------------------------------- #
# Hugging Face: index and downloads
# --------------------------------------------------------------------------- #
def models_root() -> Path:
    """Where downloaded models are kept."""
    env = os.environ.get("NGIBBS_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "ngibbs" / "models"


def hub_revision() -> str:
    """The Hugging Face revision read (``$NGIBBS_HF_REVISION``, default ``main``;
    the release workflow points it at its staging pull request)."""
    return os.environ.get("NGIBBS_HF_REVISION", "main")


def empty_index() -> dict:
    return {"format": INDEX_FORMAT, "models": {}}


_INDEX_MEMO: Dict[str, Tuple[float, dict]] = {}
_INDEX_TTL_S = 300.0


def fetch_index(*, strict: bool = False, revision: Optional[str] = None) -> dict:
    """The model index from Hugging Face. Falls back to the last copy fetched
    (with a warning) when offline, unless ``strict``."""
    rev = revision or hub_revision()
    memo = _INDEX_MEMO.get(rev)
    if memo and not strict and time.monotonic() - memo[0] < _INDEX_TTL_S:
        return memo[1]
    root = models_root()
    cached = root / INDEX_FILE
    try:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(HF_REPO_ID, INDEX_FILE, revision=rev, local_dir=root)
            index = json.loads(Path(path).read_text())
        except Exception as e:
            # (Remote)EntryNotFoundError, named differently across huggingface_hub versions
            if not any(c.__name__.endswith("EntryNotFoundError") for c in type(e).__mro__):
                raise
            index = empty_index()           # the repo exists, nothing released yet
    except Exception as e:
        if strict or not cached.is_file():
            if strict:
                raise
            warnings.warn(f"Could not reach Hugging Face ({type(e).__name__}); "
                          "only models already downloaded are available.", stacklevel=2)
            index = _index_from_disk(root)
        else:
            age = time.strftime("%Y-%m-%d", time.localtime(cached.stat().st_mtime))
            warnings.warn(f"Could not reach Hugging Face ({type(e).__name__}); "
                          f"using the model list fetched on {age}.", stacklevel=2)
            index = json.loads(cached.read_text())
    _INDEX_MEMO[rev] = (time.monotonic(), index)
    return index


def _index_from_disk(root: Path) -> dict:
    """An index of the versions already downloaded under ``root`` (offline use)."""
    index = empty_index()
    for man in root.glob(f"*/v*/{MANIFEST_FILE}"):
        try:
            m = json.loads(man.read_text())
            index["models"].setdefault(m["model"], {})[m["version"]] = m.get("index_entry", {})
        except Exception:
            continue
    return index


def version_dir(model: str, version: str) -> Path:
    return models_root() / model / f"v{version}"


def is_downloaded(model: str, version: str, entry: Optional[dict] = None) -> bool:
    """Whether ``model`` v``version`` is complete on disk (and, given its index
    entry, is the same build: a replaced pre-release counts as not downloaded)."""
    d = version_dir(model, version)
    try:
        man = json.loads((d / MANIFEST_FILE).read_text())
    except Exception:
        return False
    if entry and entry.get("fingerprint") and man.get("fingerprint") != entry["fingerprint"]:
        return False
    return all((d / rel).is_file() for rel in man.get("files", {}) if not is_on_demand(rel))


def download(model: str, version: str, entry: dict, *, verbose: bool = True) -> Path:
    """Fetch ``model`` v``version`` (everything but on-demand folders) if needed."""
    d = version_dir(model, version)
    if is_downloaded(model, version, entry):
        return d
    if d.exists():                      # stale (replaced pre-release) or partial
        shutil.rmtree(d)
    from huggingface_hub import snapshot_download
    prefix = f"{model}/v{version}"
    if verbose:
        mb = entry.get("download_bytes", 0) / 1e6
        print(f"[nGibbs] Downloading {model}v{version} ({mb:.0f} MB) from "
              f"huggingface.co/{HF_REPO_ID} to {d}")
    snapshot_download(HF_REPO_ID, revision=hub_revision(), local_dir=models_root(),
                      allow_patterns=[f"{prefix}/*"],
                      ignore_patterns=[f"{prefix}/{x}/*" for x in ON_DEMAND_DIRS])
    if not is_downloaded(model, version, entry):
        raise RuntimeError(f"Download of {model}v{version} is incomplete; try again.")
    return d


def resolve(model: str, version: Optional[str], index: dict,
            installed: Optional[str]) -> Tuple[str, dict]:
    """Pick the version of ``model`` to load: ``version`` if given (it must be
    compatible), else the newest compatible one."""
    versions = index.get("models", {}).get(model)
    if not versions:
        known = sorted(index.get("models", {}))
        raise KeyError(f"No model named {model!r} on Hugging Face ({HF_REPO_ID}). "
                       f"Available: {known or 'none'}. See ngibbs.available_models().")
    if version is not None:
        if version not in versions:
            raise KeyError(f"{model} has no version v{version}; available: "
                           f"{', '.join('v' + v for v in sorted(versions, key=version_key))}.")
        why = incompatibility(version, versions[version], installed)
        if why:
            raise RuntimeError(f"{model}v{version} {why}, but nGibbs {installed} is installed: "
                               "pip install --upgrade nGibbs")
        return version, versions[version]
    ordered = sorted(versions, key=version_key, reverse=True)
    ok = [v for v in ordered if incompatibility(v, versions[v], installed) is None]
    if not ok:
        why = incompatibility(ordered[0], versions[ordered[0]], installed)
        raise RuntimeError(f"No version of {model} works with nGibbs {installed}: "
                           f"{model}v{ordered[0]} {why}. pip install --upgrade nGibbs")
    if ok[0] != ordered[0]:
        why = incompatibility(ordered[0], versions[ordered[0]], installed)
        warnings.warn(f"{model}v{ordered[0]} is available but {why}; using {model}v{ok[0]}.",
                      stacklevel=3)
    return ok[0], versions[ok[0]]


# --------------------------------------------------------------------------- #
# public: make_ngibbs and available_models
# --------------------------------------------------------------------------- #
def _api_class(name: Optional[str]):
    from .engine.API import HeFESToAPI, MELTSAPI
    classes = {"MELTSAPI": MELTSAPI, "HeFESToAPI": HeFESToAPI}
    if name not in classes:
        raise ValueError(f"Model card api {name!r} is not one of {sorted(classes)}.")
    return classes[name]


def make_ngibbs(name: str, device: str = "cpu", *, version: Optional[str] = None,
                source: str = "auto", verbose: bool = False, **api_kwargs):
    """Load an nGibbs emulator, downloading it the first time it is used.

    Parameters
    ----------
    name : str
        A model name (``'MELTS120'``), a name with a version (``'MELTS120v1.2'``),
        or a path to a model folder of your own.
    device : str
        ``'cpu'`` or ``'cuda'``.
    version : str, optional
        Same as the ``v1.2`` suffix. Default: the newest version that works with
        the installed nGibbs (a newer one that needs an upgrade is warned about).
    source : {'auto', 'hub', 'checkout'}
        ``'hub'``: a released version from Hugging Face. ``'checkout'``: the
        working copy in a source checkout's ``TrainedModels/``. ``'auto'``
        (default): the working copy when there is one and no version is asked
        for, else the hub.
    **api_kwargs
        Passed to the API class (``MELTSAPI`` / ``HeFESToAPI``).

    The returned object records what was loaded in ``.model_name`` and
    ``.model_version`` (``'1.2'``, ``'checkout'`` or ``'custom'``).
    """
    import torch
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requested, but no CUDA device is visible.")

    path = Path(name).expanduser()
    if path.is_dir():
        bundle, model, model_version, src = path, path.name, "custom", "custom"
    else:
        model, v = parse_name(name)
        if v and version and v != version:
            raise ValueError(f"{name!r} names v{v}, but version={version!r} was also given.")
        version = version or v
        local = checkout_models()
        src = source
        if src == "auto":
            src = "checkout" if (version is None and model in local) else "hub"
        if src == "checkout":
            if model not in local:
                raise KeyError(f"No working copy of {model!r} in {CHECKOUT_MODELS_DIR}.")
            bundle, model_version = local[model], "checkout"
        elif src == "hub":
            index = fetch_index()
            model_version, entry = resolve(model, version, index, installed_version())
            bundle = download(model, model_version, entry)
        else:
            raise ValueError(f"source must be 'auto', 'hub' or 'checkout', not {source!r}.")

    card = read_card(bundle)
    if card.get("model"):
        model = card["model"]
    cls = _api_class(api_kwargs.pop("api", None) or card.get("api"))
    kwargs = dict(card.get("api_kwargs") or {})
    kwargs.update(api_kwargs)
    api = cls(bundle, device=device, verbose=verbose, **kwargs)
    api.model_name = model
    api.model_version = model_version
    api.benchmark_name = model
    if src == "checkout":
        api.benchmark_baseline_dir = baseline_dir(model, device)
    if verbose:
        print(f"[nGibbs] Loaded {model} ({model_version}) on {device} from {bundle}")
    return api


def _size(nbytes: Optional[int]) -> str:
    return f"{nbytes / 1e6:.0f} MB" if nbytes else "?"


def available_models(name: Optional[str] = None, *, as_frame: bool = False):
    """Print the models on Hugging Face.

    ``name=None``: the newest version of each model. ``'all'``: every version of
    every model. A model name (``'MELTS120'``): every version of that model.

    Columns: version, download size, whether it is already on this machine
    (``checkout``: identical to the working copy in this source checkout), and
    whether the installed nGibbs can load it. Returns the table as a pandas
    DataFrame when ``as_frame=True``.
    """
    import pandas as pd
    index = fetch_index()
    installed = installed_version()
    models = index.get("models", {})
    local = checkout_models()

    if name is None or name == "all":
        wanted = sorted(set(models) | set(local))
    else:
        wanted = [parse_name(name)[0]]
        if wanted[0] not in models and wanted[0] not in local:
            raise KeyError(f"No model named {wanted[0]!r}. Available: {sorted(set(models) | set(local))}")

    rows = []
    for model in wanted:
        versions = models.get(model, {})
        ordered = sorted(versions, key=version_key, reverse=True)
        checkout_fp = bundle_manifest(local[model])["fingerprint"] if model in local else None
        shown = ordered if (name is not None) else ordered[:1]
        for v in shown:
            e = versions[v]
            why = incompatibility(v, e, installed)
            if why and name is None:
                ok = [u for u in ordered if incompatibility(u, versions[u], installed) is None]
                why += f"; v{ok[0]} available" if ok else "; no compatible version"
            if checkout_fp and e.get("fingerprint") == checkout_fp:
                where = "checkout"
            else:
                where = "yes" if is_downloaded(model, v, e) else "no"
            rows.append({"model": model,
                         "version": f"v{v}" + (" (pre-release)" if e.get("prerelease") else ""),
                         "size": _size(e.get("download_bytes")),
                         "local": where,
                         "compatible": why or "yes"})
        if checkout_fp and not any(e.get("fingerprint") == checkout_fp for e in versions.values()):
            man = bundle_manifest(local[model])
            rows.append({"model": model, "version": "checkout (unreleased)",
                         "size": _size(sum(f["size"] for f in man["files"].values())),
                         "local": "checkout", "compatible": "yes"})

    cols = ["model", "version", "size", "local", "compatible"]
    df = pd.DataFrame(rows, columns=cols)
    header = f"nGibbs {installed or '?'}; models from huggingface.co/{HF_REPO_ID}"
    print(header)
    print(df.rename(columns={"compatible": f"compatible with nGibbs {installed or '?'}"})
          .to_string(index=False) if not df.empty else "  (no models released yet)")
    return df if as_frame else None
