"""Shared thermal data structures for the HeFESTo Python translation.

The first implementation slice uses a hybrid storage strategy:
- a row-oriented record per phase or endmember as the source of truth;
- dense property-major tensors for batched/vectorized execution.

The names in this module intentionally mirror the Fortran helper arguments so
the translation can stay traceable as more files are ported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


HCOK = 1.4387769
SIN_ASYMPTOTE_A = 23.594
SIN_ASYMPTOTE_B = 6.1
OPTIC_CONTINUUM_WIDTH_MIN = 0.01

"""ispec, apar, fn, zu, wm, To, Fo, Vo, Ko, Kop, Kopp,
               wd1, wd2, wd3, ws1, ws2, ws3,
               we1, qe1, we2, qe2, we3, qe3, we4, qe4, wou, wol,
               gam, qo, be, ge, q2A2, htl, ibv, ied, izp, Go, Gop, Got)"""

@dataclass(frozen=True)
class HeFESToThermalModeParameters:
    """Thermal mode parameters extracted from a parameter record."""

    wd1: float
    wd2: float
    wd3: float
    ws1: float
    ws2: float
    ws3: float
    we1: float
    qe1: float
    we2: float
    qe2: float
    we3: float
    qe3: float
    we4: float
    qe4: float
    wou: float
    wol: float


@dataclass(frozen=True)
class HeFESToParsetResult:
    """Structured equivalent of the Fortran ``parset`` outputs."""

    fn: float
    zu: float
    wm: float
    To: float
    Fo: float
    Vo: float
    Ko: float
    Kop: float
    Kopp: float
    modes: HeFESToThermalModeParameters
    gam: float
    qo: float
    be: float
    ge: float
    q2A2: float
    htl: float
    ibv: int
    ied: int
    izp: int
    Go: float
    Gop: float
    Got: float


@dataclass(frozen=True)
class HeFESToThermalModeTensors:
    """Batched tensor view of parset outputs for vectorized runtime use."""

    names: tuple[str, ...]
    fn: torch.Tensor
    zu: torch.Tensor
    wm: torch.Tensor
    To: torch.Tensor
    Fo: torch.Tensor
    Vo: torch.Tensor
    Ko: torch.Tensor
    Kop: torch.Tensor
    Kopp: torch.Tensor
    wd: torch.Tensor
    ws: torch.Tensor
    we: torch.Tensor
    qe: torch.Tensor
    wou: torch.Tensor
    wol: torch.Tensor
    gam: torch.Tensor
    qo: torch.Tensor
    be: torch.Tensor
    ge: torch.Tensor
    q2A2: torch.Tensor
    htl: torch.Tensor
    ibv: torch.Tensor
    ied: torch.Tensor
    izp: torch.Tensor
    Go: torch.Tensor
    Gop: torch.Tensor
    Got: torch.Tensor

    @classmethod
    def from_parset_results(
        cls,
        names: Sequence[str],
        results: Sequence[HeFESToParsetResult],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> "HeFESToThermalModeTensors":
        """Build a tensor bundle from ordered parset results."""

        if len(names) != len(results):
            raise ValueError("names and results must have the same length")

        device = torch.device("cpu") if device is None else device
        fn = torch.tensor([result.fn for result in results], dtype=dtype, device=device)
        zu = torch.tensor([result.zu for result in results], dtype=dtype, device=device)
        wm = torch.tensor([result.wm for result in results], dtype=dtype, device=device)
        To = torch.tensor([result.To for result in results], dtype=dtype, device=device)
        Fo = torch.tensor([result.Fo for result in results], dtype=dtype, device=device)
        Vo = torch.tensor([result.Vo for result in results], dtype=dtype, device=device)
        Ko = torch.tensor([result.Ko for result in results], dtype=dtype, device=device)
        Kop = torch.tensor([result.Kop for result in results], dtype=dtype, device=device)
        Kopp = torch.tensor([result.Kopp for result in results], dtype=dtype, device=device)
        wd = torch.tensor([[result.modes.wd1, result.modes.wd2, result.modes.wd3] for result in results], dtype=dtype, device=device)
        ws = torch.tensor([[result.modes.ws1, result.modes.ws2, result.modes.ws3] for result in results], dtype=dtype, device=device)
        we = torch.tensor([[result.modes.we1, result.modes.we2, result.modes.we3, result.modes.we4] for result in results], dtype=dtype, device=device)
        qe = torch.tensor([[result.modes.qe1, result.modes.qe2, result.modes.qe3, result.modes.qe4] for result in results], dtype=dtype, device=device)
        wou = torch.tensor([result.modes.wou for result in results], dtype=dtype, device=device)
        wol = torch.tensor([result.modes.wol for result in results], dtype=dtype, device=device)
        gam = torch.tensor([result.gam for result in results], dtype=dtype, device=device)
        qo = torch.tensor([result.qo for result in results], dtype=dtype, device=device)
        be = torch.tensor([result.be for result in results], dtype=dtype, device=device)
        ge = torch.tensor([result.ge for result in results], dtype=dtype, device=device)
        q2A2 = torch.tensor([result.q2A2 for result in results], dtype=dtype, device=device)
        htl = torch.tensor([result.htl for result in results], dtype=dtype, device=device)
        ibv = torch.tensor([result.ibv for result in results], dtype=torch.int64, device=device)
        ied = torch.tensor([result.ied for result in results], dtype=torch.int64, device=device)
        izp = torch.tensor([result.izp for result in results], dtype=torch.int64, device=device)
        Go = torch.tensor([result.Go for result in results], dtype=dtype, device=device)
        Gop = torch.tensor([result.Gop for result in results], dtype=dtype, device=device)
        Got = torch.tensor([result.Got for result in results], dtype=dtype, device=device)
        return cls(
            names=tuple(names),
            fn=fn,
            zu=zu,
            wm=wm,
            To=To,
            Fo=Fo,
            Vo=Vo,
            Ko=Ko,
            Kop=Kop,
            Kopp=Kopp,
            wd=wd,
            ws=ws,
            we=we,
            qe=qe,
            wou=wou,
            wol=wol,
            gam=gam,
            qo=qo,
            be=be,
            ge=ge,
            q2A2=q2A2,
            htl=htl,
            ibv=ibv,
            ied=ied,
            izp=izp,
            Go=Go,
            Gop=Gop,
            Got=Got,
        )


def resolve_thermal_parameter_directory(path: Path | str) -> Path:
    """Resolve a parameter directory and raise a clear error if it is missing."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Missing HeFESTo parameter directory: {resolved}")
    return resolved


