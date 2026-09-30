# Full project presentation

Slidev sources for the joint ETH Zurich project on implicit Hamiltonian optimal
control, liquidation applications, and the unknown-dynamics extension.

This is the full project deck. The shorter RL-focused research interview deck
is maintained separately in [presentation_m](../presentation_m/README.md).

## Run

From the repository root, in PowerShell:

```powershell
cd presentation
npm.cmd ci --include=dev --bin-links=true
npm.cmd run dev
```

Use the address printed by Slidev. For the RL section on its own:

```powershell
npm.cmd run dev:rl
```

On Linux/macOS, use `npm` instead of `npm.cmd`.

## Files

| File or directory | Purpose |
| --- | --- |
| `slides.md` | Full deck entry point, including the RL section |
| `slides-part2-rl.md` | Unknown-dynamics section |
| `script_RL.md` | Original oral notes; their older numbering is not synchronized with every slide |
| `style.css` | Shared styling |
| `components/PipelineBox.vue` | Reusable equation/pipeline component |
| `global-bottom.vue` | Continuous slide number |
| `public/images/` | Static assets served under `/images/` |
| `public/images/rl/alternate/` | Distinct older portfolio exports retained for provenance |

All active image references resolve within `public/images/`. Alternate portfolio
figures are not byte-identical to the active exports and have therefore been
retained rather than treated as duplicates. The two slide decks keep their own
assets so each presentation can be installed and built independently.

## Build

```powershell
npm.cmd run build
```

Build output goes to `dist/`, which is ignored along with `node_modules/`.
The `export` script is retained; Slidev PDF export additionally requires its
optional Playwright/Chromium dependency. The interview deck already declares
that dependency if a self-contained export workflow is needed.

## Scientific context

The slide content is retained from the full project presentation. For clarified
sign conventions, baseline architecture, test-time identification and limitations
of reported claims, consult the [interview notes](../presentation_m/README.md#conventions-et-précisions-scientifiques)
and [experiment guide](../docs/EXPERIMENTS.md). Reorganizing these files does not
constitute a revision of the underlying experiments or the report.
