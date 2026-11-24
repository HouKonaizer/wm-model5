# WM — a Self-Consistent Five-Subspace Lattice Model

An English entry point for external readers. The primary project documentation is
in Russian (`README.md`, `THEORY.md`, `EQUATIONS.md`); this file mirrors the
current state (docs date: 2026-11-23) in English.

## Positioning (honest)

WM is a **numerical** lattice platform with a self-consistent core and an
**honest audit** of every number it produces: each check is classified as a
*calibration*, a *standard formula*, a *prediction outside calibration*, or an
*honest lattice computation*. It is **not** a theory that derives α, G, m_e
from first principles — they are inputs (CODATA/PDG), as is the particle
spectrum. The QED-level agreements below are **Standard Model formulas**
evaluated through the model's own SI scale map; they are not the model's own
derivations.

## Quick start

Requires Python 3.11+, numpy.

```
pip install -e .
wm demo       # ~10 s: fast verification gate (68 checks) + English report
wm run        # full 11-phase scenario (20³) → snapshots.json + viewer.html
wm checks     # full audit: 83 checks → verification.json + audit.html
wm serve      # HTTP serving of viewer.html (file-mode frames.bin)
pytest tests  # 131 tests
```

`wm demo` is the showcase command: it runs the fast gate in ~3 s, prints the
kind counts and all predictions with deviations, and writes `verification.json`
to `demo_out/` (plus `audit.html`). The rest of the CLI prints in Russian.

`wm checks` also writes **`audit.html`** — an offline single-file dashboard of
the whole audit (4 tabs: all checks with filters/search, the 12 predictions,
frozen P1–P5, about the model). No server, no CDN, no dependencies: open the
file in a browser.

## Architecture

The world = 5 volumetric subspaces on a torus (20×20×20 = 8000 cells each):

| # | Subspace | Content |
|---|----------|---------|
| L0 | TIME | local phase and flow rate (arrow always forward; slowed in gravity wells and by speed) |
| L1 | ENERGY | energy density; strictly conserved (Noether), bounded below by the vacuum zero (Casimir) |
| L2 | PROBABILITY | complex wave function ψ (discrete Schrödinger, unitary) |
| L3 | GRAVITY | potential [−1, 0] from energy density + node self-mass; always attractive |
| L4 | FIELDS | vector field (3 components) + charge; simplified Maxwell, dipolarity |

**Vacuum = decoupling** (layers uncorrelated). **Matter = node/synthesis**: a
cell where all 5 layers are "in phase". An entropy regulator targets S = 0.35
every 10 steps (pressure P = clip(S−0.35)·0.02): S above target → "cooling"
(order, synthesis); below → "heating" (decoupling, dark-energy analog).

- **Kernel** (`wm/model.py`, `step_unified`): SSFM (split-step finite
  differences) with potential V = 2g + E; conserves node count N, momentum p,
  total energy ΣE and charge Σq to ~1e-15; the legacy bit-lock
  κ D12 = 0.4069113352735587 pins the numerics bit-for-bit.
- **SI scale map** (`wm/constants.py`): 4 anchors — U = 781.8934068 MeV
  (hadronization), α, h, dt — plus 5 free dials (G4 audit: none is derivable
  from the core). Model units: m_e = 6.535404e-4 U, ħ = 0.04933296 U·step,
  g_e = 1.861967, a₀ = 10344.26 cells, E₁ = 13.6057 eV (quantum H via the
  α-anchor, G2).
- **Experiments** (`wm/experiments.py`): all 83 checks + `verification.json`
  export + `audit.html` dashboard writer (template: `audit_template.html`).

## Verification audit (83 checks)

| Class | Count | Meaning |
|-------|-------|---------|
| Calibrations | 4 | scale anchors — identity by construction |
| Standard formulas | 50 | known formulas with SI constants through the model map; dev ≈ 0 — implementation checks, not predictions |
| Predictions outside calibration | 12 | computed from masses/anchors only, no fitting |
| Honest lattice computations | 17 | D1–D18: convergences, Noether invariants, artifact-vs-physics separation |

Key prediction numbers (model vs experiment):

| Quantity | Model | Exp | Dev |
|----------|-------|-----|-----|
| a_e (3-loop QED) | 0.001159652 | 0.001159652 | +7.06e-06 % |
| 21 cm H line | 1421.16 MHz (LO) | 1420.405751 MHz | +0.053 % |
| τ_μ (tree) | 2.18735 µs | 2.19698 µs | −0.44 % |
| BR(τ→eνν), BR(τ→μνν) (tree) | 17.78 / 17.29 % | 17.82 / 17.39 % | −0.21 / −0.55 % |
| R_π (SM, RC included) | 8095.9 | 8110.6 (PDG) | −0.18 % |
| R_K (SM, RC included) | 40371.4 | 40112.3 (KLOE 2009) | +0.65 % |
| τ(π⁰→γγ) | 84.79 fs | 84.2 fs | +0.70 % |
| τ(p-Ps→2γ) (SM series) | 125.16 ps | 124.35 ps | +0.65 % |
| n_i(Si, 300 K) | 1.17e10 cm⁻³ | 1.0e10 cm⁻³ | +17.1 % |
| κ (GPS redshift) | 0.4069 | 1.0 (GR) | −59.3 % |

SM numbers are online-verified (arXiv:0707.3439, arXiv:0907.3594,
hep-ph/9910488); RC factors and B_p are verified theoretical corrections, not
free parameters.

## Honest limitations (documented "kill" boundaries)

- **G1**: the gravity sector is a closed boundary — model G = 1.57e37 ×
  physical G; κ_g → 1.0 is unreachable without a new free parameter.
- **P1** is **falsified** at β_g = 1 (−59…−70% vs Pound–Rebka/Galileo); the
  anchor reading β_g = 1/κ is the protected form (documented as convention).
- **κ_GPS = 0.407** (−59.3% vs GR) — an honest discrepancy; the model has no
  unique weak-field limit (local template, D15a/b).
- **P5**: the model H-atom binds at 3.31e-2 eV (−99.76% vs 13.6057 eV) — a
  frozen negative boundary, not a claim.
- **21 cm**: the LO tree is kept; the full recoil tree is **HOLD**
  (coefficients not verifiable online; the naive (μ/m_e)³ factor is worse).
- **p-Ps**: SM exceeds experiment by +0.65% (tension, o-Ps-puzzle analog) —
  agreement is *not* claimed.
- All 5 dials are free parameters (G4); numerical coincidences are numerology.

## Frozen predictions (P1–P5, `PREDICTIONS.md`)

| # | Content | Status |
|---|---------|--------|
| P1 | gravitational redshift κ | falsified at β_g = 1; anchor reading β_g = 1/κ |
| P2 | κ_E/κ_g = 1/3 (dual rate channel) | partial protocol |
| P3 | vacuum-energy accretion by the well (M_eff ×2.0–2.6 / 250 steps) | partial protocol |
| P4 | w = −1 era | not a prediction (regulator diagnostics) |
| P5 | H atom | negative boundary (frozen) |

Falsification rule: |model − measurement| > 3σ of the cited experiment.

## Reproducibility

- `pytest tests` — 131 tests (unit, gates, QED precision, CLI, demo, audit).
- `wm checks` — 83 checks → `verification.json` (machine-readable audit) +
  `audit.html` (offline dashboard, this page regenerates from the same JSON).
- `wm run` — deterministic (seeded); legacy bit-lock κ D12 = 0.4069113352735587.
- GitHub repository + CI: to be added (publication stage).

## Files

| Path | What |
|------|------|
| `wm/` | the package: model (kernel), experiments (checks), constants (SI map), cli, config, particles |
| `tests/` | 131 pytest tests |
| `README.md` | primary docs (Russian), living |
| `THEORY.md` | draft axiomatics v0.11 (Russian) |
| `EQUATIONS.md` | final equation summary (Russian) |
| `ROADMAP_TO_REALITY.md` | honest audit of the 83 checks (Russian) |
| `PREDICTIONS.md` | P1–P5 protocol (frozen, Russian) |
| `PREPRINT.md` | preprint skeleton + English abstract |
| `verification.json` | machine output of the full audit (generated) |
| `audit_template.html` | audit dashboard template (marker `/*__DATA__*/null`) |
| `audit.html` (generated) | offline audit dashboard, 4 tabs (generated by `wm checks`) |
| `viewer_template.html`, `three.min.js` | 3D viewer source (offline) |

## License

MIT (`LICENSE`).
