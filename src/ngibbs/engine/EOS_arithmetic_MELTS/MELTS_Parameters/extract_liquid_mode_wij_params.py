"""Extract MAGMA's rhyolite-MELTS liquid model-parameter tables into JSON,
verbatim numeric literals, keyed by POSITION (the C code indexes each array by
position: W(n) for the n-th component/species pair (i < l, i outer), then --
1.1/1.2 only -- one H/S/V adjustment per species at NW+i):
  MELTS102: includes/param_struct_data_v34.h -> meltsModelParameters (the
            table liquid_v34.c's WH/WS/WV macros read in MODE__MELTS; 19
            components; differs from param_struct_data.h's
            originalModelParameters in the CO2 pairs)
  MELTS110: includes/param_struct_data_CO2.h -> meltsAndCO2ModelParameters
  MELTS120: includes/param_struct_data_CO2_H2O.h -> meltsAndCO2_H2OModelParameters
The pair labels in the C source are comments only; they are kept here as
'W_labels' and checked against the positional order.
Usage: python extract_liquid_mode_wij_params.py <MAGMA/includes> <out.json>"""
import json, re, sys
from pathlib import Path
SPECIES = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'MgCr2O4', 'Fe2SiO4', 'MnSi0.5O2', 'Mg2SiO4', 'NiSi0.5O2',
           'CoSi0.5O2', 'CaSiO3', 'Na2SiO3', 'KAlSiO4', 'Ca3(PO4)2', 'CO2', 'SO3', 'Cl2O-1', 'F2O-1', 'H2O', 'CaCO3']
ALIAS = {'S': 'SO3', 'Cl': 'Cl2O-1', 'F': 'F2O-1'}
TABLES = {'MELTS102': ('param_struct_data_v34.h', 'meltsModelParameters', 19, False),
          'MELTS110': ('param_struct_data_CO2.h', 'meltsAndCO2ModelParameters', 20, True),
          'MELTS120': ('param_struct_data_CO2_H2O.h', 'meltsAndCO2_H2OModelParameters', 20, True)}


def parse(path, name):
    t = Path(path).read_text()
    body = t[t.index(name):]
    body = body[body.index('{')+1: body.index('};')]
    rows = re.findall(r'\{\s*"([^"]*)"\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*[,}]', body)
    return [(lab.replace('\t', ' '), float(h), float(s), float(v)) for lab, h, s, v in rows]


out = {'_comment': __doc__.strip(), 'species': SPECIES}
for key, (fn, arr, ne, has_adjust) in TABLES.items():
    rows = parse(Path(sys.argv[1]) / fn, arr)
    nw = ne*(ne - 1)//2
    expected = nw + ne if has_adjust else nw
    if len(rows) != expected:
        raise ValueError(f"{key}: {len(rows)} entries parsed from {fn}, expected {expected}")
    pairs = [(i, l) for i in range(ne) for l in range(i+1, ne)]
    mism = []
    for n, (i, l) in enumerate(pairs):
        m = re.match(r'W\(\s*(.+?)\s*,\s*(.+?)\s*\)$', rows[n][0].strip())
        a, b = ALIAS.get(m.group(1), m.group(1)), ALIAS.get(m.group(2), m.group(2))
        if {a, b} != {SPECIES[i], SPECIES[l]}:
            mism.append((n, rows[n][0], SPECIES[i], SPECIES[l]))
    if mism:
        raise ValueError(f"{key}: pair labels disagree with positional order: {mism[:5]}")
    rec = {'species': SPECIES[:ne],
           'W': [[r[1], r[2], r[3]] for r in rows[:nw]],
           'W_labels': [r[0] for r in rows[:nw]]}
    if has_adjust:
        for i in range(ne):
            lab = rows[nw+i][0].strip()
            if ALIAS.get(lab, lab) != SPECIES[i]:
                raise ValueError(f"{key}: species adjustment {i} is {lab!r}, expected {SPECIES[i]}")
        rec['species_adjust'] = [[r[1], r[2], r[3]] for r in rows[nw:]]
    out[key] = rec
    print(key, 'entries', len(rows), 'nonzero WS/WV:', sum(1 for r in rows if r[2] != 0 or r[3] != 0),
          'nonzero species adj:', sum(1 for r in rows[nw:] if any(r[1:])) if has_adjust else 0)
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
