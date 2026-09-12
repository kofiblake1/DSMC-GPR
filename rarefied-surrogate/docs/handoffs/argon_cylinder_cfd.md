# Handoff: argon-cylinder CFD reproduction (stage [4])

**Status: not started.** This file was referenced twice by `README.md` and did
not exist (`docs/CLEANUP_TODO.md` CONFIRM #7); this resolves that by creating
the skeleton, not by doing the work.

The first milestone: reproduce the 2D argon-cylinder CFD results
(Lofthouse, Boyd & Wright 2007) with AERO-F, as the base predictor for the
surrogate.

---

## What already works, and what is not this case

The mesh → `.top` → partition → sower → AERO-F chain is **verified end to end**
on a 1 m circle. See `aerof_mesh_and_case2.md` for the recipe, the reference
numbers, and the environment traps.

**But that run is not this case.** It reproduced the FRG short-course Case2
verbatim, which means:

| | Case2 smoke test | what this case needs |
|---|---|---|
| gas | air, γ = 1.4, R = 287.1 | **argon**, γ = 5/3, monatomic |
| Prandtl | 0.72 | **2/3** (Eucken, monatomic) |
| viscosity | `Constant`, 3.82e-4 | **power law**, ω ≈ 0.734 anchored to the VHS reference |
| closure | `TurbulenceModel` + `WallFunction` | **laminar**, no turbulence model (M3, M4) |
| Re | 4.5e5 | ~10³ worst case (M4) |
| wall | adiabatic | **isothermal** at `T_wall` (see below) |

So three independent things must change in the `FluidFile`: the gas model, the
transport model, and the closure. `scripts/prep_aerof_case.sh` currently copies
the tutorial's file unmodified and warns about exactly this.

## Two things that will bite

**1. An adiabatic wall has zero heat flux by definition.** `StickMoving` maps
to `BC_ADIABATIC_WALL_MOVING` (`BcDef.h:23`). If you run with no
`SurfaceData` override and ask for `HeatFluxPerUnitSurface`, you will get
numerical noise around zero and reasonably conclude the pipeline is broken.
You need:

```
SurfaceData[1] {
  Type = Isothermal;
  Temperature = <T_wall>;
  ComputeHeatFlux = On;
}
```

The promotion happens at load time via `SubDomain::changeSurfaceType`
(`SubDomainCore.C:2914`). It matches on `face->getSurfaceID()`, which sower
reads from the text **after the last underscore** in the element-set name
(`FluidDomain.C:202-217`) — which is why `aerof/geo.py` emits
`StickMoving_1` rather than `StickMoving`.

**2. Slip must be applied afterwards, not by AERO-F.** Per M3 the base is
laminar **no-slip** NS plus a-posteriori first-order Maxwell slip. Do not try
to switch on AERO-F's built-in `MaxwellSlip`: it is dead code (empty face hook,
no callers, a default coefficient of 0.0, and no temperature-jump term at all).
The evidence is recorded under M3 in `docs/decisions/DECISIONS.md`.

## Open items

- **Solver settings** — flux, CFL law, limiter, reconstruction, and the
  supersonic startup strategy. **User-owned** (from the AERO-F tutorials), not
  a pipeline concern; see the M4 Amendment. Options are `Roe`, `VanLeer`,
  `HLLE`, `HLLC`, `RotatedRiemann`; there is no AUSM.
  <!-- FILL_IN once determined -->
- **Argon transport model** — power law with ω = 0.734 anchored to the VHS
  reference, plus Pr = 2/3. Listed in `DECISIONS.md` section C as
  *to derive/confirm*. <!-- FILL_IN -->
- **Argon constants** — molecular mass, γ = 5/3: stated but to be confirmed
  against the papers. <!-- FILL_IN -->
- **Which Kn cases** (all four vs. a focus), half vs. full cylinder symmetry,
  2-D planar confirmation. <!-- FILL_IN -->
- **Freestream conditions and `T_wall`** per Kn case, from the paper.
  <!-- FILL_IN -->
- **`size_min_factor`** — M4's wall-gradient convergence study. Run it at the
  bluntest, highest-Mach case (M4's own guidance: the stagnation station has
  the shortest gradient length). Until then the mesh falls back to
  `perimeter/n_surf`. <!-- FILL_IN -->
- **Far-field distance** — M4 wants it keyed to the shock-standoff
  correlation; the mesh currently uses a fixed `20 D`. For a cylinder at
  M ≈ 3-6 the bow shock stands off ~0.2-0.3 D, so 20 D is generous, but the
  rule is not implemented. <!-- FILL_IN -->
- **Validation targets** — which quantities and tolerances count as
  "reproduced" (surface heat flux and shear distributions vs. the paper's
  figures, presumably). <!-- FILL_IN -->

## Where the target data lives

Register the paper's digitised comparison data in `data/registry.yaml` when it
exists. The DSMC side for these Kn cases is in the `KN_eq_*` pickles — note
their provenance is still open (which SPARTA tool produced them, and their
suspiciously identical file sizes: CONFIRM #3, #5, #6).
