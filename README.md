# Wideband technosignature search: VLASS 3 GHz × Gaia DR3 nearby main-sequence stars

All code and data for the search are here: the cross-match, background and natural-emitter rejection, per-star
and population limits on 2–4 GHz emission, the LSPM J1943+5337 follow-up, and image stacking of 208,573 stars.
Built November 2026; every script was re-run end to end from this folder to verify the numbers below.

## Quick start

```bash
conda env create -f environment.yml && conda activate vlass-technosig
cd code
# Everything below runs offline from data/raw (no network needed):
python 03_sample_crossmatch.py      # ~1 min
python 07_classify_and_limits.py    # needs 05/06 outputs only for the LSPM J1943+5337 verdict (already in data/)
python 10_stack.py                  # ~35 s; holds the 1.3 GB cutout array + float64 shell copies (interactive run peaked at ~17 GB)
python 11_figures.py
python 12_compare_breakthrough_listen.py
python 13_transmitter_rate.py      # transmitter rate vs EIRP; literature points need external/seti_limits_py3 (see below)
python 14_multiepoch_detections.py # secure associations in every VLASS QL epoch 1.1-4.1 (~5 min; fetches 61x61 px cutouts from CADC)
python 15_compare_de2024.py        # overlap with De et al. 2024 (VLASS-Gaia radio stars)
python 16_compare_srsc.py          # overlap with the Sydney Radio Star Catalogue (Driessen et al. 2024)
# All-epoch analysis (needs the four cutout arrays, see 'Large files'):
python 17_forced_photometry.py     # forced photometry, every star x epoch; injection-recovery calibration (~3 min)
python 18_alleepoch_candidates.py  # candidates, confusion veto, control background, SIMBAD/X-ray (cached; --offline)
python 14_multiepoch_detections.py --candidates   # astrometry of the unexplained candidates
python 19_alleepoch_limits_stacks.py  # per-star, population and stack limits (~1 min)
python 20_alleepoch_figures.py
python 21_epoch_catalog_crosscheck.py --offline   # cross-check against per-epoch QL/SE/median-stack catalogues (~1 min)
```

Re-fetching from the archives (only needed to rebuild `data/raw`):
`01_fetch_gaia.py` (~5 min), `02_fetch_vlass_catalog.py` (~20 min), `04_fetch_counterparts.py` (~2 min),
`05_lspm1943_archival_imaging.py`, `06_lspm1943_vlass_astrometry.py` (each ~1 min), `08_cadc_wcs_table.py` (~6 min),
`08b_fill_image_dates.py` (reads DATE-OBS for the 7,722 QL images whose CAOM record lacks a date),
`09_download_cutouts.py --epochs 1.1,1.2 --tag ep1` then `--consolidate` (**~8 h** at 4 threads; please keep concurrency low;
later epochs use `--epochs 2.1,2.2 --tag ep2`, `3.1,3.2 --tag ep3`, `4.1 --tag ep4`).

Run order for a full rebuild: 01 → 02 → 02b → 03 → 04 → 05 → 06 → 07 → 08 → 08b → 09 (ep1–ep4) → 10 → 14 → 15 → 16 → 17 → 18 → 14 --candidates → 19 → 20 → 21 → 11 → 12 → 13.
`21_epoch_catalog_crosscheck.py --reduce` rebuilds its reduced catalogue from the full NRAO lists (see `data/raw` below);
`02b_e1_astrometry.py` re-derives the Epoch-1 position corrections, `vlass_e1ql_position_offsets.csv` (applied by `common.e1_components`),
from the Epoch-1 catalogue and the full QL3.1/3.2 lists in `data/raw/ql_catalogs_full/`.

## Layout

| Path | Contents |
|---|---|
| `code/common.py` | Paths, constants (per-half-epoch QL peak/integrated biases from Kimball et al. 2026: 0.85/0.90 for epoch 1.1, 0.92/1.00 later; BP−RP split 1.84, shells, service URLs), shared helpers. Also redirects astropy's config directory into the project. |
| `code/01–21_*.py` | Pipeline scripts. Each docstring lists its inputs, outputs and expected values. |
| `data/raw/` | Data exactly as fetched (see below). |
| `data/derived/` | Intermediate tables written by the scripts (sample, star–component pairs, offset pairs, reliability fit, stacked images, LSPM astrometry JSON, all-epoch forced photometry). `ms_sample_vlass.parquet` (40 MB) is not tracked: script 03 regenerates it bit-for-bit in about a minute. |
| `results/` | Figures (PNG and vector PDF) and tables. |
| `paper/` | AAS Journals manuscript (AASTeX v7.0.2), see below. |

### paper/

| File | Purpose |
|---|---|
| `ms.tex` | Manuscript for *The Astronomical Journal*. Every number in the text comes from macros in `numbers.tex`; tables are `\input` from `tables/`. The Zenodo DOI is marked `TODO`. |
| `make_paper_inputs.py` | Regenerates `numbers.tex` (142 macros), `tables/*.tex` and `figures/` from `results/` and `data/derived/`. Run after any pipeline change. |
| `build_bib.py` | Builds `refs.bib` from DOIs via CrossRef (36 entries) plus 3 manual arXiv/ASCL entries. Output is pure ASCII: BibTeX is byte-oriented and splits multi-byte UTF-8 initials such as "Ž". It also restores corporate authors (Gaia/Astropy Collaboration), maps journals to AAS abbreviations and drops month fields. `--fix-only` re-cleans an existing file offline. |
| `aastex702.cls`, `aasjournalv7.1.bst` | Official AASTeX v7.0.2 files (unmodified; full distribution in `aastex_v702_dist/`). |
| `aasjournalv7.1_patched.bst` | Copy of the AAS style with a single fix: the original leaves `next.extra = "b"` after the forward pass, which gives the last presorted reference a spurious "a" year suffix (e.g. "Ivezić et al. 2002a"). `ms.tex` uses this copy. |
| `build.sh` | Local build (pdflatex → bibtex → pdflatex ×2). Overleaf does the same automatically. |

### Literature points for the transmitter-rate figure

`13_transmitter_rate.py` draws previous surveys with `seti_limits_py3`, the plotting code of B. Jacobson-Bell
(forthcoming paper; Choza et al. 2024 correction applied). It is not redistributed here: place it in
`external/seti_limits_py3/` (git-ignored) or point `SETI_LIMITS_DIR` at it. Without it, the script still writes
`results/vlass_transmitter_rate_points.csv` and plots the VLASS curves alone.

### Large files

Every tracked file is kept below ~30 MB. Overleaf's GitHub sync rejects files over 50 MB and, through GitHub's blob API, fails above ~40 MiB.

GitHub limits files to 100 MB, so two inputs are attached to the repository's GitHub release (tag `v1.0-data`)
instead of being committed. After cloning, download them into `data/raw/` and check them with
`shasum -a 256 -c MANIFEST.sha256`:

| File | Size | Rebuild from the archives with |
|---|---|---|
| `data/raw/vlass_ep1_ql_cutouts_41px.npz` | 1.4 GB | `09_download_cutouts.py --epochs 1.1,1.2 --tag ep1` (~8 h) |
| `data/raw/vlass_ep2_ql_cutouts_41px.npz` | 1.4 GB | `09_download_cutouts.py --epochs 2.1,2.2 --tag ep2` (7.8 h) |
| `data/raw/vlass_ep3_ql_cutouts_41px.npz` | 1.4 GB | `09_download_cutouts.py --epochs 3.1,3.2 --tag ep3` (7.7 h) |
| `data/raw/vlass_ep4_ql_cutouts_41px.npz` | 0.7 GB | `09_download_cutouts.py --epochs 4.1 --tag ep4` (4.0 h) |
| `data/raw/vlass_e1ql_components.parquet` | 254 MB | `02_fetch_vlass_catalog.py` (~20 min) |

`data/raw/vlass_strips/` is a resumable download cache for script 02. It is not distributed because the
component parquet holds exactly the same data.

### data/raw

| File | Source | Size |
|---|---|---|
| `gaia_dr3_plx10_part1.parquet`, `_part2.parquet` | Gaia DR3 archive; ϖ > 10 mas, ϖ/σ > 10, δ > −45° (321,417 rows, stored as two row-halves; read with `common.read_gaia()`) | 22 + 22 MB |
| `vlass_e1ql_components.parquet` | VizieR J/ApJS/255/30/comp (Gordon et al. 2021), via the CfA mirror (3,381,277 rows) | 242 MB |
| `vlass_e1ql_subtiles.parquet` | VizieR J/ApJS/255/30/subtile (35,285 Quick Look images; observation dates, MeanIslrms) | 1.6 MB |
| `vlass_strips/` | Per-degree download cache for the component table. Lets 02 resume; deletable once the parquet exists. | 332 MB |
| `vlass_ql_wcs_cadc.parquet` | CADC CAOM per-image WCS for all 122,698 VLASS QL images (epochs 1.1–4.1) | 5.8 MB |
| `vlass_ep1_ql_cutouts_41px.npz` | 41×41 px (1″/px) Epoch-1 QL cutouts at 208,573 stars from CADC. Arrays: `star_index`, `source_id`, `cut` (Jy/beam), `off` (sub-pixel offset). Match to stars by `source_id`. | 1.3 GB |
| `vlass_ql_dates_header.csv` | DATE-OBS from the FITS headers of the 7,722 QL images without a CAOM date (mostly epoch 4.1) | — |
| `classification_overrides.csv` | Two literature reclassifications (LQ Hya, PW And: young single dwarfs), with references | — |
| `de2024_vlass_gaia_table1.csv` | VizieR J/AJ/168/288/table1 (De et al. 2024; 605 rows) | — |
| `srsc/` | Sydney Radio Star Catalogue (Driessen et al. 2024; radiostars.org): ReadMe, stars.dat, radio.dat | 1.3 MB |
| `xray_counterparts.csv` | ROSAT 2RXS (40″) and eRASS1 (20″) nearest counterparts for the 73 candidate stars (VizieR mirror) | — |
| `simbad_candidates.csv`, `simbad_nearest.csv` | SIMBAD (CfA mirror) identifications and object types | — |
| `vlass_ql_catalogs_near_stars.parquet` | Components within 60″ of any sample star from the NRAO per-epoch lists, de-duplicated (cat = QL3.1, QL3.2, QL4.1: interim QL lists, VLASS Memo 22; QLMED: QL median stack of Epochs 1–3; SE2.1–SE3.2: Single Epoch Continuum Imaging lists, 2026-06 versions). Position errors in on-sky arcsec. Built by `21 --reduce` | 7.4 MB |
| `vlass_catalog_images.parquet` | Every image (QL subtile / SE pointing) present in each of those lists, with its component count; defines the partial SE sky coverage | 0.3 MB |
| `vlass_e1ql_position_offsets.csv` | Per-subtile Epoch-1 minus Epoch-3 QL position offsets (w-term; Kimball et al. 2026) from 242,942 compact S/N > 20 pairs, with tile and declination fallbacks; written by `02b_e1_astrometry.py` | 2.1 MB |
| `ql_catalogs_full/` | *Not tracked* (0.2–1.3 GB each). The full lists, downloaded 2026-11-13 from the NRAO SharePoint folder `VLASS_catalogs/{QL,SECI}` (M. Lacy) linked from science.nrao.edu/vlass while vlass-dl.nrao.edu was offline: `QL3.1_components.fits`, `QL3.2_components.fits`, `QL4.1_components_v1.fits`, `QL_medstack_components_interim.fits`, `SECI{2.1,2.2,3.2}_components_20260601.fits`, `SECI3.1_components_20260620.fits`. A CIRADA Epoch-2 QL csv placed here is picked up automatically. | — |
| `lspm1943/` | DSS POSS-I/II and PS1 cutouts, PS1 calibration stars, Gaia neighbours, and VLASS QL/SE multi-extension cutouts (target plus 5 references) from CADC | 24 MB |

## Key results (as reproduced by these scripts)

- **Sample:** 299,746 Gaia sources with BP−RP → 218,654 main sequence (−1.5 < ΔM_G < +1.0 about the ridge) → 210,191 in the VLASS footprint.
- **Epoch-1 astrometry:** Epoch-1 QL positions lack the w-term correction (Kimball et al. 2026). Per-subtile median offsets from 242,942 compact
  S/N > 20 Epoch-1/QL3 pairs (ΔDec ≈ +0.42″ at δ < −20° to −0.13″ at δ > 60°) are removed before matching.
- **Positional model** (proper motion and parallax): σ_i² = (k·σ_cat)² + σ_corr² + σ_sys² with k = 1.50, σ_sys = 0.04″ per axis;
  50 genuine stellar detections; **51 matches with reliability > 0.5** (48 with R > 0.9; 1.5 expected false). V376 Cep drops to R = 0.01
  (corrected Epoch-1 offset 0.65″ vs 0.14″ in QL3.1) and is treated as confusion-limited.
- **Classification of the 51** (SIMBAD types first; multiplicity and activity evidence recorded separately):
  - 9 flare stars, 6 RS CVn, 6 BY Dra, 4 young stars, 2 CVs, 1 magnetic CP star;
  - 20 multiple or over-luminous stars without a more specific type, 2 X-ray-active stars;
  - 1 radio source not coincident with its star (LSPM J1943+5337).
  - 30 of the 51 are in the Sydney Radio Star Catalogue (11 circularly polarized); 41 are in De et al. (2024).
- **Multi-epoch (VLASS 1.1–4.1, 177 independent images):** 44/51 variable; 21 co-moving with their star, 17 inconclusive, 12 single-epoch, none stationary.
- **LSPM J1943+5337:** the radio emission is not at the star: χ² = 77.9 for 3 dof in declination, 104 for 6 dof in RA+Dec
  (VLASS 1.2/2.2/3.2; frame tied to Gaia within 0.03″). Stationary vs co-moving at a ~0.8″ offset is not yet decided; RA positions
  scatter beyond their errors (source extended E–W). Archival plates show no optical counterpart (POSS-I g ≳ 20.4, r ≳ 19.7–19.9; POSS-II ≳ 20.9).
- **Forced photometry, all epochs (720,108 star-epochs; median 3 per star):** peak-loss factor κ = 0.861 from injection–recovery
  (catalogue cross-check 0.899 ± 0.010, a 4% systematic); injected sources are recovered at 5σ in 51% of cases, as Φ(S/σ − 5) predicts.
- **Candidates:** 160 stars at ≥ 5σ (coadd or any epoch), 143 with X-ray/SIMBAD activity evidence. Steady test (coadd): 7 unexplained
  vs 11.4 expected at control positions → μ95 = 5.5. Single-epoch test: 16 vs 11.8 → μ95 = 12.8. The one candidate bright enough for
  per-epoch astrometry (TYC 3573-1391-1) lies 1.8″ from its star.
- **Cross-check with per-epoch catalogues (21):** κ from the QL3.1–4.1 lists 0.83–0.87 (adopted 0.861). Our bias-corrected fluxes are
  1.13 ± 0.01 × the Single Epoch peaks (1.07–1.16 by epoch), and SE/QL = 0.975 ± 0.006 for the same Epoch-3 sources: faint compact QL
  peaks are less biased than the calibrator-based 0.92, so later-epoch fluxes and limits may be ≈13% conservative. 90% (SE) / 67% (QL) of forced ≥5σ star-epochs have a catalogue match. Of the 17
  unexplained candidates, G 133-48, MCC 401 and HD 317101A are detected at the star in independent catalogues (MCC 401 and
  LP 198-661 also in the median stack); TYC 3573-1391-1 sits 1.7–1.8″ off in four catalogues. An own-epoch confusion veto gives
  7→10 vs b 11.4→23.4 (steady); the union of both vetoes changes μ95 by −4% (steady) and +11% (single-epoch). The median stack
  (152,511 low-motion stars) finds no steady emitters missed by the coadd (9 unclassified matches, all 1.9–2.3″ off; 22 chance).
- **Per-star limits (95%, all epochs, steady):** median 0.54 mJy, L_ν < 4.0×10⁸ W Hz⁻¹; Barnard's star L_ν < 2.3×10⁵ W Hz⁻¹
  (Epoch-1 catalogue analysis: median 1.09 mJy; Barnard 5.1×10⁵).
- **Population (steady):** f < 8.0×10⁻³ / 2.4×10⁻⁴ / 2.6×10⁻⁵ above L_ν = 10⁷ / 10⁸ / 10⁹ W Hz⁻¹; per snapshot f·δ < 1.8×10⁻⁵.
  (Epoch-1 catalogue, with no unexplained associations by construction: 2.1×10⁻² / 6.4×10⁻⁴ / 2.2×10⁻⁵, floor 1.4×10⁻⁵.)
- **Stacks (all epochs):** M-dwarf mean > 4σ in 4 of 6 shells, 20–40 pc: (7.9 ± 1.1)×10⁵ W Hz⁻¹ (7.5σ); median S/σ shift including
  detected stars 0.084 ± 0.013 (6.2σ). AFGK ≤ 3.8σ. Mean stacks reach f < 1 at L_T ≈ 1.3×10⁵ (AFGK) and 1.6×10⁵ (M) W Hz⁻¹ (nearest
  shells); median stacks give f < 0.09–0.10 at 10⁷ and 0.015–0.028 at 10⁹ W Hz⁻¹.
- **Versus Breakthrough Listen (Price et al. 2020):**
  - VLASS (all epochs) is about 13,100× deeper in flux density.
  - For narrowband lines it is about 1.5×10⁵× less sensitive in EIRP.
  - A search of the same GBT data matched to bandwidth B would lose sensitivity only as √B, so the continuum advantage is coverage
    (signals the turboSETI pipeline does not search for) and sample size, not depth.

## Reproducibility notes

- **Bit-identical on rerun:** everything downstream of `data/raw`. Script 03 sorts the Gaia table by `source_id` before the seeded offset realisations.
- **Differs slightly from the original session:**
  1. The offset-control realisation. The Gaia archive returned rows in a different order, so seed 1 drew different position angles. The background density is 330.5 offset pairs within 10″ here vs 334.3 originally, and expected false matches are 2.46 vs 2.49. The set of secure matches was identical (before the 2026 astrometry/flux-bias update).
  2. The DSS/PS1 blank-sky apertures in 05 are a different random draw. The 3σ plate limits move by ≤ 0.13 mag; the SNRs at the radio position are unchanged.
- **Live services:** SIMBAD is a live database, so object types can change. The tables used here are frozen in `data/raw`. When CDS or CIRADA were unreachable (Nov 2026), the CfA VizieR and SIMBAD mirrors and CADC served the same data.
- **CADC cutout syntax:** CADC ignores `?CIRCLE=`. Use pixel `?SUB=[x1:x2,y1:y2,*,*]` (1-based); several `SUB` boxes in one request return one FITS extension per box.

## Conventions and caveats

- **Flux and limits:**
  - QL peak fluxes and image noise are divided by b_pk = 0.85 (epoch 1.1) or 0.92 (later); Epoch-1 catalogue integrated fluxes by 0.90 (1.1) or 1.00 (Kimball et al. 2026).
  - Limits for non-detected stars are (5 + 1.645) × the subtile MeanIslrms.
  - EIRP = L_ν × 2 GHz is the total in-band power of a steady signal of any width that survives RFI flagging and imaging.
- **Duty cycle:** each QL position is observed for only a few seconds, so the fractions are fraction × duty cycle.
- **Stack interpretation:**
  - Excluding individually detected stars truncates each shell at a luminosity threshold proportional to d². That is why the M-dwarf mean rises with distance.
  - Each shell is used only for transmitter powers below its 5th-percentile per-star limit.
  - M-dwarf stack limits are conservative because natural emission is present.

## Citation of data

Gaia DR3 (Gaia Collaboration 2023); VLASS (Lacy et al. 2020) and the CIRADA Epoch 1 QL catalogue (Gordon et al. 2021,
ApJS 255, 30); NRAO interim QL and median-stack component lists (Lacy, Dong et al. 2025, VLASS Memo 22) and Single Epoch
component lists (NRAO); ROSAT 2RXS (Boller et al. 2016); eRASS1 (Merloni et al. 2024); SIMBAD (Wenger et al. 2000);
DSS (STScI); Pan-STARRS1 (Chambers et al. 2016; Flewelling et al. 2020); CADC; VizieR (Ochsenbein et al. 2000).
