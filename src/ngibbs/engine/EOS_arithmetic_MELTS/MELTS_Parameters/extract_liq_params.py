"""
Extract MAGMA's `includes/liq_struct_data.h` `Liquid meltsLiquid[] = {...}`
table (19 liquid oxide/silicate components, rhyolite-MELTS mode) into JSON.

Struct layout (from the header comments in liq_struct_data.h itself):

    {"label", NULL,
        {                                 -- ThermoRef (solid reference state)
            H_ref, S_ref, V_ref,
            CP_TYPE, {{k0,k1,k2,k3,Tt,dH,l1,l2}},
            EOS_TYPE, {{v1,v2,v3,v4}}
        },
        {                                 -- ThermoLiq (liquid-specific)
            V_liq,
            EOS_KRESS, {{dvdt, dvdp, d2vdtp, d2vdp2}},
            T_fusion, S_fusion, Cp_liquid, T_glass
        }
    },

We only need the ThermoLiq block for the Kress volume EOS (melts_vec's
liquid_eos.py) -- V_liq and the four Kress coefficients, plus Cp_liquid
(a flat constant per MAGMA, not a Berman polynomial) and T_fusion/S_fusion
for completeness. The ThermoRef block (solid H/S/V/Cp/EOS at Tr,Pr) is also
captured since a couple of MAGMA's special-cased liquid components (notably
H2O) use their own hardcoded volume formula rather than this table's Kress
coefficients -- flagged separately in liquid_eos.py, not derived from this
JSON.

Reuses the same brace/quote-aware tokenizer as extract_sol_params.py rather
than a real C parser -- sufficient because this file has the same very
regular, machine-generated structure.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path


def strip_comments(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == '\\' and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == '/' and i + 1 < n and text[i + 1] == '*':
            j = text.find('*/', i + 2)
            i = n if j == -1 else j + 2
            continue
        if c == '/' and i + 1 < n and text[i + 1] == '/':
            j = text.find('\n', i)
            i = n if j == -1 else j
            continue
        out.append(c)
        i += 1
    return ''.join(out)


def find_matching_brace(text: str, open_idx: int) -> int:
    assert text[open_idx] == '{'
    depth = 0
    i = open_idx
    in_str = False
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if c == '\\':
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("no matching brace")


def split_top_level(text: str, sep: str = ',') -> list[str]:
    parts = []
    depth = 0
    in_str = False
    cur = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            cur.append(c)
            if c == '\\':
                i += 1
                if i < n:
                    cur.append(text[i])
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            cur.append(c)
        elif c in '{(':
            depth += 1
            cur.append(c)
        elif c in ')}':
            depth -= 1
            cur.append(c)
        elif c == sep and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        parts.append(''.join(cur))
    return [p.strip() for p in parts]


_NUM_OK = re.compile(r'^[0-9eE+\-.*/() ]+$')


def parse_double(tok: str) -> float:
    tok = tok.strip()
    try:
        return float(tok)
    except ValueError:
        pass
    if _NUM_OK.match(tok):
        return float(eval(tok, {"__builtins__": {}}, {}))
    raise ValueError(f"could not parse double: {tok!r}")


def parse_number_list(brace_text: str) -> list[float]:
    inner = brace_text.strip()
    assert inner.startswith('{') and inner.endswith('}')
    inner = inner[1:-1].strip()
    if inner.startswith('{') and inner.endswith('}'):
        inner = inner[1:-1].strip()
    if not inner:
        return []
    return [parse_double(t) for t in split_top_level(inner)]


def parse_entry(entry_text: str) -> dict:
    # entry_text is the inside of the outer {...} for one Liquid record,
    # i.e. `"label", NULL, {ThermoRef...}, {ThermoLiq...}`
    parts = split_top_level(entry_text)
    label = parts[0].strip().strip('"')

    # ThermoRef block is parts[2] (a brace group); ThermoLiq is parts[3].
    # Re-locate them robustly by brace position rather than trusting the
    # comma-split (nested braces are already respected by split_top_level).
    ref_text = parts[2].strip()
    liq_text = parts[3].strip()

    ref_fields = split_top_level(ref_text.strip()[1:-1])
    h = parse_double(ref_fields[0])
    s = parse_double(ref_fields[1])
    v = parse_double(ref_fields[2])
    cp_type = ref_fields[3].strip()
    cp_coeffs = parse_number_list(ref_fields[4])
    eos_type = ref_fields[5].strip()
    eos_coeffs = parse_number_list(ref_fields[6])

    liq_fields = split_top_level(liq_text.strip()[1:-1])
    v_liq = parse_double(liq_fields[0])
    kress_type = liq_fields[1].strip()
    kress_coeffs = parse_number_list(liq_fields[2])
    t_fusion = parse_double(liq_fields[3])
    s_fusion = parse_double(liq_fields[4])
    cp_liquid = parse_double(liq_fields[5])
    t_glass = parse_double(liq_fields[6])

    return dict(
        label=label,
        ref_h=h, ref_s=s, ref_v=v,
        ref_cp_type=cp_type, ref_cp_coeffs=cp_coeffs,
        ref_eos_type=eos_type, ref_eos_coeffs=eos_coeffs,
        v_liq=v_liq, kress_type=kress_type, kress_coeffs=kress_coeffs,
        t_fusion=t_fusion, s_fusion=s_fusion,
        cp_liquid=cp_liquid, t_glass=t_glass,
    )


def extract_table(src: str, table_name: str) -> list[dict]:
    marker = f"Liquid {table_name}[] = {{"
    start = src.find(marker)
    if start == -1:
        raise KeyError(f"table {table_name!r} not found")
    open_idx = src.index('{', start)
    close_idx = find_matching_brace(src, open_idx)
    body = src[open_idx + 1:close_idx]

    entries = []
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == '{':
            j = find_matching_brace(body, i)
            entries.append(body[i + 1:j])
            i = j + 1
        else:
            i += 1
    return [parse_entry(e) for e in entries]


def main():
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/mnt/user-data/uploads/MAGMA/includes/liq_struct_data.h")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "liq_struct_data.json"

    text = strip_comments(src_path.read_text())

    tables = {}
    for name in ("xMeltsLiquid", "meltsLiquid", "meltsFluidLiquid", "pMeltsLiquid"):
        try:
            tables[name] = extract_table(text, name)
        except KeyError:
            print(f"  (table {name!r} not found -- skipping)")
        except ValueError as e:
            # xMeltsLiquid uses calibration-multiplier expressions
            # (e.g. "-906377.0 *SIO2_MULT + corrH01") that reference
            # #define'd constants elsewhere in the file -- out of scope
            # (we only need meltsLiquid, the rhyolite-MELTS default mode).
            print(f"  (table {name!r} failed to parse: {e} -- skipping, not needed)")

    with open(out_path, 'w') as fh:
        json.dump(tables, fh, indent=2)

    for name, records in tables.items():
        print(f"{name}: {len(records)} records")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
