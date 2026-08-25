# Component identity: the root fix

Short answer: **the right convention already exists in your codebase — it just stops at the
projection matrix.** `compToOxV2.csv` keys 136 of its 147 rows as `"<species> : <phase>"`,
including both magnetites:

```
magnetite : spinel
magnetite : ferropericlase
high-pressure-magnetite : ca-ferrite
```

So the lowest layer already says a component is identified by the *pair*. The bug is that
the two naming registries above it don't, and one of them invented a second, incompatible
convention.

I'd argue against the rename, and propagate this one instead.

---

## Why MELTS survives and HeFESTo doesn't

Your diagnosis is right, and the numbers are worse than you'd expect. Colliding component
names in the current registries:

| database | slots | distinct names | species in >1 phase |
|---|---|---|---|
| HeFESTo | 158 | 76 | **1** (`magnetite`) |
| MELTS | 236 | 72 | **15** |

MELTS collides *far* more — `diopside`, `clinoenstatite`, `hedenbergite`, `jadeite`,
`alumino-buffonite`, `buffonite`, `essenite` across opx/cpx; `albite`, `anorthite`,
`sanidine` across four feldspar phases; `leucite`, `analcime`, `na-leucite` across two.

It survives because the MELTS importer reads **phase blocks** — the phase is structurally
present at every read, so the species name never has to carry it. The HeFESTo importer
reads `fort.99`, which is a **flat species list with no phase labels**, so it must
reconstruct the phase from a name. That reconstruction is the only place identity has to be
recovered rather than carried, and it is exactly where it broke.

## The specific inelegance

`HEFESTO_ABBREVIATION_TO_SHORT_NAMES` used a hyphen suffix to disambiguate:

```python
'smag': 'magnetite-spinel'
```

But the hyphen suffix is *already* how genuine mineral names are spelled in that table —
`mg-wadsleyite`, `fe-ringwoodite`, `ferric-bridgmanite`, `mg-akimotoite`, and nine more all
end in `-<phase>` and are *not* disambiguated; that is simply their name. `magnetite-spinel`
is the only entry where the suffix means something different.

So the two cases are **indistinguishable by shape**, which is why my first patch — strip
`-<phase>` if the remainder is a schema key — was a heuristic that happened to work rather
than a rule. It should not survive.

## What I changed

**1. `constants.py` — promote the existing convention.**

```python
COMPONENT_KEY_SEP = ' : '
def split_component_key(key): ...        # 'magnetite : spinel' -> ('magnetite', 'spinel')

'smag': 'magnetite : spinel',
'mag':  'magnetite : ferropericlase',
```

No mineral is renamed. MELTS is untouched. The disambiguation is written in the form the
data files already use, and resolution becomes a `split`, not a guess:

```
smag   magnetite : spinel           spinel           -> 'magnetite'      col 34
mag    magnetite : ferropericlase   ferropericlase   -> 'magnetite'      col 143
mgwa   mg-wadsleyite                wadsleyite       -> 'mg-wadsleyite'  col 60
```

`reconcile_component_name` now also **refuses a contradiction**: asked to write
`'magnetite : spinel'` into `ferropericlase`, it returns `None` rather than falling through
to the bare name.

**2. `indexer._look_for_illegal_oxides` — a second instance of the same bug, one layer
down.** It read the composite key and then threw the phase away:

```python
comp_name = comp_full.split(' : ')[0].strip()
for phase, components in self.MELTS_indices.items():
    if comp_name in components:
        self.EXCLUDED_COMPONENTS_BY_PHASE[phase].add(comp_name)   # every phase!
```

One row describes one `(species, phase)` pair, but an exclusion triggered by
`magnetite : spinel`'s oxide set would also drop magnetite from ferropericlase — and for
MELTS, one bad `diopside : orthopyroxene` row would take out cpx's diopside too. Now
filtered by `key_phase`.

**3. `DatasetIndexer.validate_component_registry()` — the part that makes this not recur.**

Three invariants, checked at construction, cheap:

- every `(phase, species)` the schema exposes has a stoichiometry row, composite or bare;
- every abbreviation HeFESTo actually emits resolves to a species that is a column
  *somewhere* — and, when the key is composite, a column of *that* phase;
- the abbreviation map is injective, so two abbreviations cannot land on one column.

Verified both ways:

```
FIXED registry:    0 problem(s)
PRE-FIX registry:  1 problem(s)
   'smag' -> 'magnetite-spinel' is not a component of ANY phase;
             its values would be silently discarded at import
```

That middle invariant is the one that matters. It needs no control file, no data, and no
knowledge of which phase `smag` belongs to — it only asks whether the resolved name could
*ever* be written anywhere. Had it existed, the fault would have been a one-line startup
error instead of years of quietly deleted spinel assemblages.

## On renaming to `fmag`

Your instinct about *which* one to move was right — spinel's magnetite is shared with
MELTS, so only the ferropericlase one could safely move. Two reasons I'd still not do it:

- **`HeFESTo_snames_short` is the join key against `fort.99`'s header**, which literally
  prints `mag`. Renaming that entry breaks the join. Only the *long* name is nGibbs's to
  choose, and the composite key is a better choice for it than a new abbreviation.
- It fixes the case, not the class. The next collision — a new HeFESTo phase, or MELTS
  gaining a species that already exists elsewhere — reopens it. The validator closes it.

If you still want the rename for readability, it now costs nothing: `'mag': 'fmag :
ferropericlase'` would fail the validator immediately unless the schema column is renamed
too, which is the point.

## Follow-ups worth considering

- **Run the validator in `DatasetIndexer.__init__`** (`strict=True` in CI, warn in
  interactive use). I left it opt-in so it can't break a running pipeline unexpectedly;
  wiring it in is a one-line change and I'd do it.
- **The `-<phase>` suffix ambiguity is still latent** in the thirteen genuine names
  (`mg-wadsleyite` and friends). They're correct today, but nothing stops a future entry
  from meaning the other thing. Moving *all* disambiguation to `" : "` and reserving the
  hyphen for real mineral names would make that structural.
- **MELTS is exposed to the same class of fault** the moment anything there reads a flat
  species list — its 15 collisions are simply never asked to resolve a phase from a name
  today. The validator covers both databases for invariant 1.
