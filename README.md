# Wideband technosignature search: VLASS 3 GHz × Gaia DR3 nearby main-sequence stars

All code and data for the search are here: the cross-match, background and natural-emitter rejection, per-star
and population limits on 2–4 GHz emission, the LSPM J1943+5337 follow-up, and image stacking of 208,573 stars.
Built November 2026; every script was re-run end to end from this folder to verify the numbers below.

## Quick start

```bash
conda env create -f environment.yml && conda activate vlass-technosig
cd code
# Everything below 03 runs offline from data/raw (no network needed):
python 03_sample_crossmatch.py      # ~1 min
python 07_classify_and_limits.py    # needs 05/06 outputs only for the LSPM J1943+5337 verdict (already in data/)
python 10_stack.py                  # ~35 s; holds the 1.3 GB cutout array + float64 shell copies (interactive run peaked at ~17 GB)
python 11_figures.py
python 12_compare_breakthrough_listen.py
```

Re-fetching from the archives (only needed to rebuild `data/raw`):
`01_fetch_gaia.py` (~5 min), `02_fetch_vlass_catalog.py` (~20 min), `04_fetch_counterparts.py` (~2 min),
`05_lspm1943_archival_imaging.py`, `06_lspm1943_vlass_astrometry.py` (each ~1 min), `08_cadc_wcs_table.py` (~6 min),
`09_download_cutouts.py --epochs 1.1,1.2 --tag ep1` then `--consolidate` (**~8 h** at 4 threads; please keep concurrency low).

Run order for a full rebuild: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12.

## Layout

| Path | Contents |
|---|---|
| `code/common.py` | Paths, constants (QL flux bias 0.87, BP−RP split 1.84, shells, service URLs), shared helpers. Also redirects astropy's config directory into the project. |
| `code/01–12_*.py` | Pipeline scripts. Each docstring lists its inputs, outputs and expected values. |
| `data/raw/` | Data exactly as fetched (see below). |
| `data/derived/` | Intermediate tables written by the scripts (sample, star–component pairs, offset pairs, reliability fit, stacked images, LSPM astrometry JSON). |
| `results/` | Figures (PNG and vector PDF) and tables. |
| `paper/` | AAS Journals manuscript (AASTeX v7.0.2), see below. |

### paper/

| File | Purpose |
|---|---|
| `ms.tex` | Manuscript for *The Astronomical Journal*. Every number in the text comes from macros in `numbers.tex`; tables are `\input` from `tables/`. Author block, acknowledgments, UAT keyword IDs, facility keywords and the Zenodo DOI are marked `TODO`. |
| `make_paper_inputs.py` | Regenerates `numbers.tex` (102 macros), `tables/*.tex` and `figures/` from `results/` and `data/derived/`. Run after any pipeline change. |
| `build_bib.py` | Builds `refs.bib` from DOIs via CrossRef (36 entries) plus 3 manual arXiv/ASCL entries. Output is pure ASCII: BibTeX is byte-oriented and splits multi-byte UTF-8 initials such as "Ž". It also restores corporate authors (Gaia/Astropy Collaboration), maps journals to AAS abbreviations and drops month fields. `--fix-only` re-cleans an existing file offline. |
| `aastex702.cls`, `aasjournalv7.1.bst` | Official AASTeX v7.0.2 files (unmodified; full distribution in `aastex_v702_dist/`). |
| `aasjournalv7.1_patched.bst` | Copy of the AAS style with a single fix: the original leaves `next.extra = "b"` after the forward pass, which gives the last presorted reference a spurious "a" year suffix (e.g. "Ivezić et al. 2002a"). `ms.tex` uses this copy. |
| `build.sh` | Local build (pdflatex → bibtex → pdflatex ×2). Overleaf does the same automatically. |

### Large files

GitHub limits files to 100 MB, so two inputs are attached to the repository's GitHub release (tag `v1.0-data`)
instead of being committed. After cloning, download them into `data/raw/` and check them with
`shasum -a 256 -c MANIFEST.sha256`:

| File | Size | Rebuild from the archives with |
|---|---|---|
| `data/raw/vlass_ep1_ql_cutouts_41px.npz` | 1.4 GB | `09_download_cutouts.py` (~8 h) |
| `data/raw/vlass_e1ql_components.parquet` | 254 MB | `02_fetch_vlass_catalog.py` (~20 min) |

`data/raw/vlass_strips/` is a resumable download cache for script 02. It is not distributed because the
component parquet holds exactly the same data.

### data/raw

| File | Source | Size |
|---|---|---|
| `gaia_dr3_plx10.parquet` | Gaia DR3 archive; ϖ > 10 mas, ϖ/σ > 10, δ > −45° (321,417 rows) | 42 MB |
| `vlass_e1ql_components.parquet` | VizieR J/ApJS/255/30/comp (Gordon et al. 2021), via the CfA mirror (3,381,277 rows) | 242 MB |
| `vlass_e1ql_subtiles.parquet` | VizieR J/ApJS/255/30/subtile (35,285 Quick Look images; observation dates, MeanIslrms) | 1.6 MB |
| `vlass_strips/` | Per-degree download cache for the component table. Lets 02 resume; deletable once the parquet exists. | 332 MB |
| `vlass_ql_wcs_cadc.parquet` | CADC CAOM per-image WCS for all 122,698 VLASS QL images (epochs 1.1–4.1) | 5.8 MB |
| `vlass_ep1_ql_cutouts_41px.npz` | 41×41 px (1″/px) Epoch-1 QL cutouts at 208,573 stars from CADC. Arrays: `star_index`, `source_id`, `cut` (Jy/beam), `off` (sub-pixel offset). Match to stars by `source_id`. | 1.3 GB |
| `xray_counterparts.csv` | ROSAT 2RXS (40″) and eRASS1 (20″) nearest counterparts for the 73 candidate stars (VizieR mirror) | — |
| `simbad_candidates.csv`, `simbad_nearest.csv` | SIMBAD (CfA mirror) identifications and object types | — |
| `lspm1943/` | DSS POSS-I/II and PS1 cutouts, PS1 calibration stars, Gaia neighbours, and VLASS QL/SE multi-extension cutouts (target plus 5 references) from CADC | 24 MB |

## Key results (as reproduced by these scripts)

- **Sample:** 299,746 Gaia sources with BP−RP → 218,654 main sequence (−1.5 < ΔM_G < +1.0 about the ridge) → 210,191 in the VLASS footprint.
- **Positional model:** σ_sys = 0.231″ per axis; 48.3 genuine stellar detections; **50 matches with reliability > 0.5** (2.46 expected false).
- **Classification of the 50:**
  - 37 active binaries: 22 binary/over-luminous + X-ray, 8 RS CVn, 6 BY Dra, 1 binary without X-ray;
  - 7 flare or X-ray-active single stars (3 flare stars, 4 X-ray active);
  - 5 other natural emitters: 2 young stars, 2 CVs, 1 magnetic CP star;
  - 1 background source (LSPM J1943+5337).

  **No unexplained secure match remains.**
- **LSPM J1943+5337:** the radio source is stationary across VLASS 1.2/2.2/3.2. The model with the radio source at the star's position gives χ² = 85.7 for 3 dof (frame tied to Gaia within 0.11″). Archival plates show no optical counterpart (POSS-I g ≳ 20.4, r ≳ 19.7–19.9; POSS-II ≳ 20.9).
- **Per-star limits (95%):** median 1.11 mJy, giving L_ν < 8.3×10⁸ W Hz⁻¹. The best is Barnard's star, L_ν < 5.4×10⁵ W Hz⁻¹ (EIRP ≈ 1.1×10¹⁵ W if the signal fills 2 GHz).
- **Population (individual non-detections):** f < 2.1×10⁻² / 6.5×10⁻⁴ / 2.2×10⁻⁵ above L_ν = 10⁷ / 10⁸ / 10⁹ W Hz⁻¹, and f < 1.4×10⁻⁵ above about 3×10⁹ W Hz⁻¹.
- **Stacking (Epoch 1):** natural sub-threshold M-dwarf emission is detected (20–40 pc: 11.6 ± 1.9 µJy, 6.2σ; L_ν ≈ 1.2×10⁶ W Hz⁻¹). FGK stacks are ≤ 2.6σ. The stacked limits on fraction × power reach f < 1 at L_T ≈ 2.65×10⁵ (FGK) and 2.43×10⁵ (M) W Hz⁻¹, 4–6× fainter than the individual non-detections.
- **Versus Breakthrough Listen (Price et al. 2020):**
  - VLASS is about 6,300× deeper in flux density.
  - For narrowband lines it is about 3.2×10⁵× less sensitive in EIRP.
  - It is more constraining for signals wider than about 300 kHz.

## Reproducibility notes

- **Bit-identical on rerun:** everything downstream of `data/raw`. Script 03 sorts the Gaia table by `source_id` before the seeded offset realisations.
- **Differs slightly from the original session:**
  1. The offset-control realisation. The Gaia archive returned rows in a different order, so seed 1 drew different position angles. The background density is 330.5 offset pairs within 10″ here vs 334.3 originally, and expected false matches are 2.46 vs 2.49. The set of 50 secure matches is identical.
  2. The DSS/PS1 blank-sky apertures in 05 are a different random draw. The 3σ plate limits move by ≤ 0.13 mag; the SNRs at the radio position are unchanged.
- **Live services:** SIMBAD is a live database, so object types can change. The tables used here are frozen in `data/raw`. When CDS or CIRADA were unreachable (Nov 2026), the CfA VizieR and SIMBAD mirrors and CADC served the same data.
- **CADC cutout syntax:** CADC ignores `?CIRCLE=`. Use pixel `?SUB=[x1:x2,y1:y2,*,*]` (1-based); several `SUB` boxes in one request return one FITS extension per box.

## Conventions and caveats

- **Flux and limits:**
  - QL fluxes are divided by 0.87 to correct their known underestimate.
  - Limits for non-detected stars are (5 + 1.645) × the subtile MeanIslrms.
  - EIRP = L_ν × 2 GHz, valid only for a flat signal filling the band. A narrowband line is diluted across the band.
- **Duty cycle:** each QL position is observed for only a few seconds, so the fractions are fraction × duty cycle.
- **Stack interpretation:**
  - Excluding individually detected stars truncates each shell at a luminosity threshold proportional to d². That is why the M-dwarf mean rises with distance.
  - Each shell is used only for transmitter powers below its 5th-percentile per-star limit.
  - M-dwarf stack limits are conservative because natural emission is present.

## Citation of data

Gaia DR3 (Gaia Collaboration 2023); VLASS (Lacy et al. 2020) and the CIRADA Epoch 1 QL catalogue (Gordon et al. 2021,
ApJS 255, 30); ROSAT 2RXS (Boller et al. 2016); eRASS1 (Merloni et al. 2024); SIMBAD (Wenger et al. 2000);
DSS (STScI); Pan-STARRS1 (Chambers et al. 2016; Flewelling et al. 2020); CADC; VizieR (Ochsenbein et al. 2000).
