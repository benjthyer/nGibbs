#!/usr/bin/env python3
"""Search BurnMan for components used by the HeFESTo emulator.

Creates a readable report of which emulator phases are present in
burnman (SLB_2024/SLB_2022), lists endmember names/formulae, and
searches for emulator component names and shorthand variants.

Run from the repo root (or an environment where `src` is importable).
"""
import sys
from pathlib import Path
import traceback


def setup_paths():
    try:
        repo_root = Path(__file__).resolve().parent.parent
    except Exception:
        repo_root = Path.cwd()
    src_root = repo_root / "src"
    nmelts_root = src_root / "nMELTS"
    for p in (str(src_root), str(nmelts_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo_root


def main():
    repo_root = setup_paths()

    try:
        import burnman
    except Exception as e:
        print("ERROR: could not import burnman:", e)
        return 1

    try:
        from nMELTS.config.constants import HEFESTO_ABBREVIATION_TO_SHORT_NAMES
        from nMELTS.engine.API import HeFESToEmulatorCPU
    except Exception as e:
        print("ERROR: could not import nMELTS helpers (run from repo root where src is available):", e)
        return 1

    shorthand_map = HEFESTO_ABBREVIATION_TO_SHORT_NAMES or {}

    emulator = HeFESToEmulatorCPU
    if emulator is None:
        print("Pre-instantiated emulator `HeFESToEmulatorCPU` is not available. Create an API instance first.")
        return 1

    ml_indexer = emulator.isentropic_emulator.model.ml_indexer
    compnames = list(ml_indexer.label_names)
    label_indices = dict(ml_indexer.label_indices)

    def _ensure_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    for mineral, idx in label_indices.items():
        try:
            print("\n" + "#" * 60)
            print(f"EMULATOR PHASE: {mineral} (index={idx})")
            # Handle scalar or iterable idx (numpy array, torch tensor, list, etc.)
            try:
                if isinstance(idx, int):
                    idxs = [int(idx)]
                else:
                    # Try to iterate (works for lists, tuples, numpy arrays, torch tensors)
                    idxs = [int(i) for i in idx]
            except Exception:
                try:
                    # Some tensor types support .item()
                    idxs = [int(idx.item())]
                except Exception:
                    idxs = []
            MinComps = [compnames[i] for i in idxs if 0 <= i < len(compnames)]
            print("  Emulator components:", MinComps)

            bm_obj = None
            found_ver = None
            for ver in ("SLB_2024", "SLB_2022"):
                module = getattr(burnman.minerals, ver, None)
                if module and hasattr(module, mineral):
                    try:
                        bm_obj = getattr(module, mineral)()
                        found_ver = ver
                        break
                    except Exception:
                        bm_obj = None

            if bm_obj is None:
                print("  -> Not found in burnman SLB_2024/2022")
                continue

            print(f"  -> Found in burnman.minerals.{found_ver}")
            em_names = getattr(bm_obj, "endmember_names", []) or []
            em_forms = getattr(bm_obj, "endmember_formulae", []) or []
            formula = getattr(bm_obj, "formula", None)
            if em_names:
                print("    endmember_names:", em_names)
            if em_forms:
                print("    endmember_formulae:", em_forms)
            if formula:
                print("    formula:", formula)

            for comp in MinComps:
                print(f"    Searching for component '{comp}':")
                found_any = False
                candidates = {str(comp)}
                # add shorthand(s) if mapping contains the token
                if str(comp) in shorthand_map:
                    candidates.add(shorthand_map[str(comp)])
                for k, v in shorthand_map.items():
                    if str(comp) == v:
                        candidates.add(k)

                for cand in list(candidates):
                    cand = str(cand)
                    if any(cand == str(n) or cand in str(n) for n in em_names):
                        print(f"      - matched endmember name: {cand}")
                        found_any = True
                    if any(cand == str(f) or cand in str(f) for f in em_forms):
                        print(f"      - matched endmember formula: {cand}")
                        found_any = True
                    if formula and cand in str(formula):
                        print(f"      - matched phase formula: {cand}")
                        found_any = True
                if not found_any:
                    print("      - No match in this burnman phase")

        except Exception:
            print(f"ERROR processing emulator phase {mineral}:")
            traceback.print_exc()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
