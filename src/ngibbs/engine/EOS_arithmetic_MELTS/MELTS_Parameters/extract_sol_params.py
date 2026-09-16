#!/usr/bin/env python3
"""
Extract endmember ThermoRef records from MAGMA's sol_struct_data.h into JSON.

sol_struct_data.h holds several `Solids <name>[] = { ... };` C initializer
arrays (one per MELTS calculation mode: xMeltsSolids, meltsSolids,
meltsFluidSolids, pMeltsSolids). Each array element is a `Solids` struct
(see includes/silmin.h) whose relevant fields for our purposes are:

    { label, type, formula, inclInClb, inStdSet, solToOx, solToLiq,
      mw, nAtoms,
      { h, s, v,                                  <- ThermoRef header
        cp_type, {k0,k1,k2,k3,Tt,deltah,l1,l2},    <- Cp model (Berman/Saxena)
        eos_type, {eos coefficients} },            <- EOS model (Berman/Vinet/Saxena)
      ... (ThermoData 'cur', na, nr, function pointers -- only present for
           PHASE entries with na>1; we don't need these)
    }

This script does a small brace/quote-aware tokenizer (not a full C parser)
to pull out label/type/formula/mw/nAtoms and the ThermoRef sub-struct for
every entry whose H/S/V aren't the all-zero dummy placeholder used for
solid-solution PHASE headers (those get their properties from mixing
models, not directly from a ThermoRef).
"""
import json
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/mnt/user-data/uploads/MAGMA/includes/sol_struct_data.h")
TABLES = sys.argv[2].split(",") if len(sys.argv) > 2 else \
    ["xMeltsSolids", "meltsSolids", "meltsFluidSolids", "pMeltsSolids"]


def strip_comments(text: str) -> str:
    """Remove /* ... */ and // ... comments (outside of quoted strings)."""
    out = []
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == '"' and text[i - 1] != '\\':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif text[i:i+2] == "/*":
            j = text.find("*/", i + 2)
            i = (j + 2) if j != -1 else n
        elif text[i:i+2] == "//":
            j = text.find("\n", i + 2)
            i = j if j != -1 else n
        else:
            out.append(c)
            i += 1
    return "".join(out)


def find_matching_brace(text: str, open_idx: int) -> int:
    """Given index of an opening '{', return index of its matching '}'."""
    depth = 0
    i = open_idx
    n = len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            if c == '"' and text[i - 1] != '\\':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"No matching brace found starting at {open_idx}")


def split_top_level(text: str) -> list:
    """Split a comma-separated field list, respecting nested {}/() and strings."""
    fields = []
    depth = 0
    in_str = False
    buf = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            buf.append(c)
            if c == '"' and text[i - 1] != '\\':
                in_str = False
        elif c == '"':
            in_str = True
            buf.append(c)
        elif c in "{(":
            depth += 1
            buf.append(c)
        elif c in ")}":
            depth -= 1
            buf.append(c)
        elif c == ',' and depth == 0:
            fields.append("".join(buf).strip())
            buf = []
        else:
            buf.append(c)
        i += 1
    if buf:
        fields.append("".join(buf).strip())
    return [f for f in fields if f != ""]



# Symbols we treat as "#define"d for the build we're extracting (rhyolite-
# MELTS / MODE__MELTS, matching the meltsSolids table and the rhm-oxide
# solution model already confirmed to key off this same symbol in
# rhomsghiorso.c). Only affects a couple of #ifdef RHYOLITE_ADJUSTMENTS /
# #else blocks in this file (e.g. SANIDINE_ADJUSTMENT).
_ACTIVE_SYMBOLS = {"RHYOLITE_ADJUSTMENTS"}


def resolve_macros(text: str) -> str:
    """Minimal line-oriented C-preprocessor pass: resolves single-line
    `#define NAME value` / `#undef NAME` blocks (including the handful
    that are guarded by `#ifdef RHYOLITE_ADJUSTMENTS`/`#else`/`#endif`)
    by literal substitution, so numeric fields that reference a locally
    #define'd calibration constant (e.g. `-3970791.0+(SANIDINE_ADJUSTMENT)`
    for sanidine, or the `H31/S31/V31` etc. scratch constants used for a
    few clinopyroxene endmembers) parse as plain arithmetic instead of
    being silently dropped by parse_entry's `except ValueError: return
    None`.

    This is NOT a general preprocessor (no macro arguments, no nested
    #if expressions) -- sol_struct_data.h only uses this pattern in a
    few small, flat blocks, confirmed by inspection.
    """
    lines = text.split("\n")
    out_lines = []
    macros: dict[str, str] = {}
    # Stack of (branch_taken: bool) for nested #ifdef/#ifndef; we only
    # ever see one level of nesting in this file, but handle a stack for
    # safety.
    cond_stack: list[bool] = []

    define_re = re.compile(r'^\s*#\s*define\s+(\w+)\s+(.+?)\s*$')
    undef_re = re.compile(r'^\s*#\s*undef\s+(\w+)\s*$')
    ifdef_re = re.compile(r'^\s*#\s*ifdef\s+(\w+)\s*$')
    ifndef_re = re.compile(r'^\s*#\s*ifndef\s+(\w+)\s*$')
    else_re = re.compile(r'^\s*#\s*else\s*$')
    endif_re = re.compile(r'^\s*#\s*endif\b')

    def active() -> bool:
        return all(cond_stack)

    for line in lines:
        m = ifdef_re.match(line)
        if m:
            cond_stack.append(m.group(1) in _ACTIVE_SYMBOLS)
            continue
        m = ifndef_re.match(line)
        if m:
            cond_stack.append(m.group(1) not in _ACTIVE_SYMBOLS)
            continue
        if else_re.match(line):
            if cond_stack:
                cond_stack[-1] = not cond_stack[-1]
            continue
        if endif_re.match(line):
            if cond_stack:
                cond_stack.pop()
            continue

        if not active():
            continue  # inside a not-taken #ifdef/#else branch: drop the line

        m = define_re.match(line)
        if m:
            name, value = m.group(1), m.group(2).strip()
            # Only track simple object-like macros with a numeric-ish
            # value (skip include guards like `_Sol_Struct_Data_h`, which
            # has no trailing value anyway and won't match define_re's
            # `(.+?)` group as empty).
            macros[name] = f"({value})"
            continue
        m = undef_re.match(line)
        if m:
            macros.pop(m.group(1), None)
            continue
        if line.lstrip().startswith("#"):
            continue  # other directives (include guards, #ifdef __cplusplus, etc.)

        if macros:
            for name, value in macros.items():
                line = re.sub(rf'\b{re.escape(name)}\b', value, line)
        out_lines.append(line)

    return "\n".join(out_lines)


_ARITH_OK = re.compile(r"^[0-9eE+\-.*/() ]+$")


def parse_double(tok: str) -> float:
    tok = tok.strip()
    try:
        # C float literals: handle trailing/embedded forms like 1.0E-6, -0.784E-6
        return float(tok)
    except ValueError:
        # Source occasionally writes an inline sum, e.g. "311.29+303.909-297.499"
        # instead of a single literal. Safe to arithmetic-eval: characters are
        # restricted to digits/operators/parens only (no names, no calls).
        if not _ARITH_OK.match(tok):
            raise
        return float(eval(tok, {"__builtins__": {}}, {}))


def parse_number_list(brace_expr: str) -> list:
    """'{{a, b, c}}' or '{a, b, c}' -> [a, b, c] as floats."""
    inner = brace_expr.strip()
    while inner.startswith("{") and inner.endswith("}"):
        stripped = inner[1:-1].strip()
        # only unwrap one layer if the whole thing is exactly one nested group
        if stripped.startswith("{") and stripped.endswith("}") and \
           find_matching_brace(stripped, 0) == len(stripped) - 1:
            inner = stripped
            continue
        inner = stripped
        break
    return [parse_double(x) for x in split_top_level(inner)]


def parse_entry(entry_body: str) -> dict | None:
    """entry_body is the text strictly inside one element's outer { }."""
    fields = split_top_level(entry_body)
    if len(fields) < 10:
        return None  # not a well-formed Solids initializer

    label = fields[0].strip().strip('"')
    type_ = fields[1].strip()
    formula = fields[2].strip().strip('"')
    mw = fields[7].strip()
    natoms = fields[8].strip()

    thermo_ref_expr = fields[9]
    # thermo_ref_expr looks like: { h, s, v, CP_TYPE, {{...8 nums...}}, EOS_TYPE, {{...n nums...}} }
    inner = thermo_ref_expr.strip()
    assert inner.startswith("{") and inner.endswith("}"), (label, inner[:40])
    inner = inner[1:-1].strip()
    ref_fields = split_top_level(inner)
    if len(ref_fields) < 7:
        return None

    try:
        h = parse_double(ref_fields[0])
        s = parse_double(ref_fields[1])
        v = parse_double(ref_fields[2])
    except ValueError as e:
        print(f"  DROPPED entry {label!r} (type={type_}): {e}", file=sys.stderr)
        return None

    if h == 0.0 and s == 0.0 and v == 0.0:
        return None  # dummy placeholder (solid-solution PHASE header)

    cp_type = ref_fields[3].strip()
    cp_coeffs = parse_number_list(ref_fields[4])
    eos_type = ref_fields[5].strip()
    eos_coeffs = parse_number_list(ref_fields[6])

    return dict(
        label=label, type=type_, formula=formula,
        mw=mw, natoms=natoms,
        h=h, s=s, v=v,
        cp_type=cp_type, cp_coeffs=cp_coeffs,
        eos_type=eos_type, eos_coeffs=eos_coeffs,
    )


def extract_table(text: str, table_name: str) -> list:
    marker = re.search(rf"\bSolids\s+{re.escape(table_name)}\s*\[\s*\]\s*=\s*", text)
    if not marker:
        raise ValueError(f"table {table_name!r} not found")
    open_brace = text.index("{", marker.end())
    close_brace = find_matching_brace(text, open_brace)
    body = text[open_brace + 1:close_brace]

    entries = []
    i = 0
    n = len(body)
    while i < n:
        if body[i] == "{":
            end = find_matching_brace(body, i)
            entry = parse_entry(body[i + 1:end])
            if entry is not None:
                entries.append(entry)
            i = end + 1
        else:
            i += 1
    return entries


def main():
    raw = SRC.read_text()
    clean = strip_comments(raw)
    clean = resolve_macros(clean)

    result = {}
    for table in TABLES:
        entries = extract_table(clean, table)
        result[table] = entries
        print(f"{table}: {len(entries)} endmember records "
              f"(eos types: "
              f"{sorted(set(e['eos_type'] for e in entries))})", file=sys.stderr)

    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
