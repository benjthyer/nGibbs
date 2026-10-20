"""
Cute little converter: oxide wt% -> oxide mol% and element mol%.

Edit COMPOSITION below (or pass a dict to `convert`) and run:
    python scripts/oxide_to_moles.py

Element fractions are reported two ways:
  * cation basis (no O, closed to 1) -- matches emulator.convertOxToMol
  * all atoms including oxygen
"""
import re

COMPOSITION = {  # wt%, Here Shergotty: McSween and Stolper 
    'SiO2': 50.4, 'TiO2': 0.81, 'Al2O3': 6.89, 'FeO': 18.4, 'Fe2O3': 0.8,
    'MgO': 9.27, 'CaO': 10.1, 'Na2O': 1.37, 'K2O': 0, 'P2O5': 0,
    'H2O': 0, 'Cr2O3': 0.21, 'MnO': 0, 'NiO': 0, 'CO2': 0.0,
}

ATOMIC_MASS = {
    'H': 1.008, 'C': 12.011, 'O': 15.999, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'K': 39.098, 'Ca': 40.078,
    'Ti': 47.867, 'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845, 'Ni': 58.693,
}


def parse_formula(formula):
    """'Al2O3' -> {'Al': 2, 'O': 3}"""
    atoms = {}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        if el not in ATOMIC_MASS:
            raise ValueError(f"Unknown element '{el}' in {formula}")
        atoms[el] = atoms.get(el, 0) + (int(n) if n else 1)
    return atoms


def molar_mass(formula):
    return sum(ATOMIC_MASS[el] * n for el, n in parse_formula(formula).items())


def convert(wt_pct):
    ox_moles = {ox: w / molar_mass(ox) for ox, w in wt_pct.items() if w > 0}

    el_moles = {}
    for ox, mol in ox_moles.items():
        for el, n in parse_formula(ox).items():
            el_moles[el] = el_moles.get(el, 0.0) + n * mol

    def close(d):
        tot = sum(d.values())
        return {k: v / tot for k, v in d.items()}

    cations = {el: m for el, m in el_moles.items() if el != 'O'}
    return close(ox_moles), close(cations), close(el_moles)


def show(title, fracs, emoji):
    print(f"\n{emoji}  {title}")
    print("  " + "-" * 32)
    for k, v in sorted(fracs.items(), key=lambda kv: -kv[1]):
        bar = "█" * round(v * 40)
        print(f"  {k:>6}  {24 * v:8.4f} %  {bar}")
    print(f"  {'sum':>6}  {24 * sum(fracs.values()):8.4f} %")


if __name__ == '__main__':
    total = sum(COMPOSITION.values())
    print(f"🪨  Input oxides: {total:.2f} wt% total")
    ox, cat, allel = convert(COMPOSITION)
    show("Oxide mole %", ox, "🧪")
    show("Element mole % (cation basis, no O)", cat, "⚛️ ")
    show("Element mole % (including O)", allel, "💨")
    print()
