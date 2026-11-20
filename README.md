# Wideband technosignature search: VLASS 3 GHz × Gaia DR3 nearby main-sequence stars

All code and data for the search are here: the cross-match, background and natural-emitter rejection, per-star
and population limits on 2–4 GHz emission, the LSPM J1943+5337 follow-up, and all-epoch forced photometry and stacking
(208,573 Epoch-1 cutouts; usable forced photometry in at least one epoch for 208,232 stars).
Built November 2026; every script was re-run end to end from this folder to verify the numbers below.

Archived at Zenodo, doi:[10.5281/zenodo.24326437](https://doi.org/10.5281/zenodo.24326437) (reserved; the record is
published when the paper is accepted). The Zenodo record also holds the large inputs from the `v1.0-data` release.

## Quick start

```bash
conda env create -f environment.yml && conda activate vlass-technosig
cd code
# Runs from data/raw without network access, except the steps marked [net]; their outputs are already in results/ and
# data/derived/, so an offline run can skip them and use the supplied tables.
python 03_sample_crossmatch.py      # ~1 min
python 03b_bright_star_check.py     # [net: SIMBAD TAP] bright (V<4) nearby dwarfs missing from the Gaia DR3 sample
python 03d_gcns_comparison.py       # completeness and contamination against the Gaia Catalogue of Nearby Stars (cached)
python 04b_activity_all.py          # activity evidence and technosignature eligibility for every star: 2RXS/eRASS1 (HEASARC TAP), SIMBAD types, Gaia DR3 EBs (cached)
python 07_classify_and_limits.py    # needs 05/06 outputs only for the LSPM J1943+5337 verdict (already in data/)
python 10_stack.py                  # ~35 s; holds the 1.3 GB cutout array + float64 shell copies (interactive run peaked at ~17 GB)
python 11_figures.py
python 14_multiepoch_detections.py # [net unless data/raw/multiepoch_detections/ is present] secure associations in every QL epoch 1.1-4.1
python 15_compare_de2024.py        # overlap with De et al. 2024 (VLASS-Gaia radio stars)
python 16_compare_srsc.py          # overlap with the Sydney Radio Star Catalogue (Driessen et al. 2024)
# All-epoch analysis (needs the four cutout arrays, see 'Large files'):
python 17_forced_photometry.py     # forced photometry, every star x epoch; injection-recovery calibration (~3 min)
python 03c_bright_dwarfs_forced.py  # needs 17; forced photometry of the 03b bright dwarfs at their SIMBAD positions (CADC cutouts, cached)
python 18_alleepoch_candidates.py  # needs 04b; forced detections, confusion veto, control background, SIMBAD/X-ray (cached; --offline)
python 14_multiepoch_detections.py --candidates   # astrometry of the forced detections
python 19_alleepoch_limits_stacks.py  # per-star, population and stack limits (~3 min)
python 20_alleepoch_figures.py
python 12_compare_breakthrough_listen.py   # needs 18 and 19
python 13_transmitter_rate.py      # needs 19; literature points need external/seti_limits_py3 (see below)
python 21_epoch_catalog_crosscheck.py --offline   # cross-check against per-epoch QL/SE/median-stack catalogues (~1 min)
python 22_cutout_astrometry.py     # needs 21 and the image headers cached by 14; per-epoch positions and co-motion tests (~3 min)
python 23_cosmos_flux_scale.py     # needs 17; sub-threshold flux scale on VLA-COSMOS 3 GHz sources (CADC cutouts; per-image results cached)
cd ../paper && python make_paper_inputs.py && sh build.sh   # macros, tables and figures for the manuscript, then the PDF
```

Re-fetching from the archives (only needed to rebuild `data/raw`):
`01_fetch_gaia.py` (~5 min), `02_fetch_vlass_catalog.py` (~20 min), `04_fetch_counterparts.py` (~2 min),
`05_lspm1943_archival_imaging.py`, `06_lspm1943_vlass_astrometry.py` (each ~1 min), `08_cadc_wcs_table.py` (~6 min),
`08b_fill_image_dates.py` (reads DATE-OBS for the 7,722 QL images whose CAOM record lacks a date),
`09_download_cutouts.py --epochs 1.1,1.2 --tag ep1` then `--consolidate` (**~8 h** at 4 threads; please keep concurrency low;
later epochs use `--epochs 2.1,2.2 --tag ep2`, `3.1,3.2 --tag ep3`, `4.1 --tag ep4`).

Run order for a full rebuild: 01 → 02 → 02b → 03 → 03b → 03d → 04 → 04b → 05 → 06 → 07 → 08 → 08b → 09 (ep1–ep4) → 10 → 11 → 14 → 15 → 16 →
17 → 03c → 18 → 14 --candidates → 19 → 20 → 12 → 13 → 21 → 22 → 23 → paper/make_paper_inputs.py.
`21_epoch_catalog_crosscheck.py --reduce` rebuilds its reduced catalogue from the full NRAO lists (see `data/raw` below);
`02b_e1_astrometry.py` re-derives the Epoch-1 position corrections, `vlass_e1ql_position_offsets.csv` (applied by `common.e1_components`),
from the Epoch-1 catalogue and the full QL3.1/3.2 lists in `data/raw/ql_catalogs_full/`.

## Layout

| Path | Contents |
|---|---|
| `code/common.py` | Paths, constants (per-half-epoch QL peak/integrated biases from Kimball et al. 2026: 0.85/0.90 for epoch 1.1, 0.92/1.00 later; BP−RP split 1.84, shells, service URLs), shared helpers. Also redirects astropy's config directory into the project. |
| `code/01–23_*.py` | Pipeline scripts. Each docstring lists its inputs, outputs and expected values. |
| `data/raw/` | Data exactly as fetched (see below). |
| `data/derived/` | Intermediate tables written by the scripts (sample, star–component pairs, offset pairs, reliability fit, stacked images, LSPM astrometry JSON, all-epoch forced photometry). `ms_sample_vlass.parquet` (40 MB) is not tracked: script 03 regenerates it bit-for-bit in about a minute. |
| `results/` | Figures (PNG and vector PDF) and tables. |
| `paper/` | AAS Journals manuscript (AASTeX v7.0.2), see below. |

### paper/

| File | Purpose |
|---|---|
| `ms.tex` | Manuscript for *The Astronomical Journal*. Every number in the text comes from macros in `numbers.tex`; tables are `\input` from `tables/`. The Zenodo DOI is marked `TODO`. |
| `make_paper_inputs.py` | Regenerates `numbers.tex` (~600 macros), `tables/*.tex` (seven tables) and `figures/` from `results/` and `data/derived/`. Run after any pipeline change. |
| `build_bib.py` | Builds `refs.bib` from DOIs via CrossRef (74 entries) plus 7 manual arXiv/ASCL/journal entries. Output is pure ASCII: BibTeX is byte-oriented and splits multi-byte UTF-8 initials such as "Ž". It also restores corporate authors (Gaia/Astropy Collaboration), maps journals to AAS abbreviations and drops month fields. `--fix-only` re-cleans an existing file offline. |
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
| `xray_counterparts.csv` | ROSAT 2RXS (40″) and eRASS1 (20″) nearest counterparts for the 77 stars that had an Epoch-1 component within 2.5″ before the w-term astrometric correction (75 after it; the paper uses those 75) (VizieR mirror) | — |
| `simbad_candidates.csv`, `simbad_nearest.csv` | SIMBAD (CfA mirror) identifications and object types | — |
| `xray_2rxs_flux.parquet`, `xray_erass1_flux.parquet` | Positions and fluxes of all 135,118 2RXS (count rate) and 930,203 eRASS1 Main (0.2–2.3 keV flux) sources (HEASARC TAP `rass2rxs`, `erass1main`), matched at the survey epochs for the activity evidence and log L_X/L_bol of every star (04b) | see MANIFEST |
| `gaia_dr3_eclipsing_binaries_100pc.parquet` | Gaia DR3 `vari_eclipsing_binary` sources with parallax > 10 mas (Mowlavi et al. 2023), an excluded class in the eligibility (04b) | see MANIFEST |
| `simbad_otypes_all.parquet` | SIMBAD activity object types (X, Er\*, Fl\*, BY\*, RS\*, TT\*, Y\*O, CV\*, a2\*, Em\*) of all objects with a Gaia DR3 identifier and parallax > 9 mas (04b) | 0.5 MB |
| `gcns_dec41.parquet` | Gaia Catalogue of Nearby Stars (Smart et al. 2021; Gaia archive `external.gaiaedr3_gcns_main_1`), δ > −41°, for the completeness check (03d) | 13 MB |
| `bright_dwarfs_simbad_pm.csv`, `bright_dwarfs_cutouts.npz` | SIMBAD positions, proper motions and parallaxes of the 03b bright dwarfs, and their 41″ QL cutouts from CADC (03c) | 0.8 MB |
| `vla_cosmos_3ghz.csv` | VLA-COSMOS 3 GHz catalogue (Smolčić et al. 2017; HEASARC `vlacos3ghz`), 10,830 sources (23) | 0.9 MB |
| `cosmos_forced_parts/` | Per-image forced photometry of the compact, isolated VLA-COSMOS sources in QL half-epochs 1.1, 2.1, 3.1 (23; lets the script resume) | — |
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
- **Classification of the 51** (secure associations only, R > 0.5; SIMBAD types first; an association is explained only by
  activity evidence -- a 2RXS/eRASS1 counterpart, a SIMBAD X-ray-source type (X), or a SIMBAD flare/BY Dra/RS CVn/
  emission-line (Em*)/young/CV/magnetic CP type -- the same rule as for the forced detections; multiplicity is recorded
  separately and explains nothing):
  - 9 flare stars, 6 RS CVn, 6 BY Dra, 4 young stars, 2 CVs, 1 magnetic CP star;
  - 21 X-ray-active stars without a more specific type (19 with multiplicity evidence);
  - 1 emission-line star: 2MASS J17022950+1341015 (M dwarf, 30.5 pc; Epoch 1 only, forced S/N 4.9 then <2: a flare);
  - 1 radio source not coincident with its star (LSPM J1943+5337); in the all-epoch tests it is treated like any other star
    (an M dwarf below L_env, so not counted), since the control-position background counts such coincidences.
  - 30 of the 51 are in the Sydney Radio Star Catalogue (11 circularly polarized); 41 are in De et al. (2024).
- **Multi-epoch (VLASS 1.1–4.1, 177 independent images):** 44/51 variable; 21 co-moving with their star, 17 inconclusive, 12 single-epoch, none stationary.
- **LSPM J1943+5337:** the radio emission is not at the star: χ² = 77.9 for 3 dof in declination, 104 for 6 dof in RA+Dec
  (VLASS 1.2/2.2/3.2; frame tied to Gaia within 0.03″). Stationary vs co-moving at a ~0.8″ offset is not yet decided; RA positions
  scatter beyond their errors (source extended E–W). Archival plates show no optical counterpart (POSS-I g ≳ 20.4, r ≳ 19.7–19.9; POSS-II B_J ≳ 21.1, R_F ≳ 20.9; at the corrected Epoch-1 position).
- **Forced photometry, all epochs (720,108 star-epochs; median 3 per star):** peak-loss factor κ = 0.861 from injection–recovery
  (catalogue cross-check 0.899 ± 0.010, a 4% systematic); injected sources are recovered at 5σ in 51% of cases, as Φ(S/σ − 5) predicts.
- **Technosignature eligibility (04b; replaces the earlier exclusion of every star with activity evidence, which removed
  73% of the stars within 10 pc but 9% at 80–100 pc):** a detection counts as a possible transmitter
  - at any luminosity at the 32,985 *eligible* stars: AFGK (BP−RP < 1.84), not a SIMBAD CV/RS CVn/young/magnetic CP star or a Gaia DR3
    eclipsing binary (133 in the sample), log L_X/L_bol < −4 (eRASS1 flux, else 2RXS count rate × 6e-12; L_bol from M_G), and no Gaia
    multiplicity evidence (RUWE > 1.4, NSS solution, ΔM_G < −0.5);
  - only above L_env = 10¹⁰ W Hz⁻¹ at the 174,955 *envelope* stars: 161,290 M dwarfs and 13,665 AFGK stars with multiplicity evidence
    (above the DG CVn superflare, 4.5×10⁹ W Hz⁻¹, Fender et al. 2015, and every stellar detection in VLASS, max 4.5×10⁹);
  - never at the 2,251 stars of the excluded classes or with log L_X/L_bol ≥ −4.
  Under the presence rule only 5 of 44 AFGK stars within 10 pc were eligible (11%, vs 82% at 80–100 pc); now 64% vs 67%.
  The X-ray constraint is complete within 20 pc; 44% of AFGK stars at 80–100 pc are unconstrained (paper Table, tables/xray_depth.tex).
  Presence of activity evidence is kept only for the astrophysical stack split. Uncatalogued multiples remain eligible and are counted in k
  (conservative).
- **Forced detections** (the paper's term; scripts and files still say 'candidates'): 161 stars at ≥ 5σ (coadd or any epoch),
  149 with activity evidence (137 with a 2RXS/eRASS1 counterpart, 12 by SIMBAD types alone). 90 are M dwarfs (all below L_env;
  e.g. G 133-48, a rapid rotator with steady L_ν ≈ 4.6×10⁷ W Hz⁻¹ and cutout positions favoring the star), 60 are AFGK stars of the
  excluded classes or with saturated coronae, and 11 are at AFGK stars outside them (Table 3): 8 with Gaia multiplicity evidence
  (12 Sco B, V335 Peg, HD 114993, δ Cap, 71 Tau, 109 Vir, HD 317101A, V2082 Cyg; none reaches L_env) and 3 at eligible stars, counted in k:
  TYC 3573-1391-1 (steady, 7.8σ; emission 1.7–1.8″ off the star), 1 Boo and TYC 7253-259-1 (single-epoch, 5.0–5.5σ).
  Steady test (coadd): k = 1 vs b = 1.3 (bootstrap ± 0.4; per control angle 0–3) → μ95 = 4.0, or 4.7 without background
  subtraction (adopted). Single-epoch test: 3 vs 2.3 (P(n ≥ k) = 0.39) → μ95 = 5.8; host-level (duty cycle): 3 stars vs 1.9 → μ95 = 6.1.
  With the AFGK multiples eligible at all luminosities: 7 vs 1.6 (P = 1×10⁻³) and 10 vs 2.8 (P = 6×10⁻⁴), the excess that motivated the
  multiplicity criterion. X-ray threshold −3.5: 2 vs 1.3 / 4 vs 2.3; −4.5: 1 vs 1.3 / 3 vs 2.1.
- **Cross-check with per-epoch catalogues (21):** κ from the QL3.1–4.1 lists 0.83–0.87 (adopted 0.861). Our bias-corrected fluxes are
  1.13 ± 0.01 × the Single Epoch peaks (1.07–1.16 by epoch), and SE/QL = 0.975 ± 0.006 for the same Epoch-3 sources: faint compact QL
  peaks are less biased than the calibrator-based 0.92, so later-epoch fluxes and limits may be ≈13% conservative. 90% (SE) / 67% (QL) of forced ≥5σ star-epochs have a catalogue match. G 133-48 is
  detected at the star in an independent catalogue; TYC 3573-1391-1 sits 1.7–1.8″ off in four catalogues; LSPM J1943+5337 is in two.
  An own-epoch confusion veto gives 1→2 vs b 1.3→3.0 (steady; P(n ≤ k) = 0.42, per angle 0–5); the union of both vetoes:
  1 vs 1.3 (steady) and 3 vs 2.1 (single-epoch, P = 0.36). The median stack
  (152,511 low-motion stars) finds no steady emitters missed by the coadd (9 unclassified matches, all 1.9–2.3″ off; 22 chance).
- **Per-star limits (95%, all epochs, steady):** median 0.54 mJy, L_ν < 4.0×10⁸ W Hz⁻¹; Barnard's star L_ν < 2.3×10⁵ W Hz⁻¹
  (Epoch-1 catalogue analysis: median 1.09 mJy; Barnard 5.1×10⁵).
- **Population (steady; eligible stars, plus envelope stars above L_env; p_i = 0 at the excluded stars):** headline, no
  background subtracted (the 1 steady detection counted, t_i = 5σ_i, or max(5σ_i, S_env,i) at envelope stars): f < 5.3×10⁻² / 1.3×10⁻³ /
  1.5×10⁻⁴ above L_ν = 10⁷ / 10⁸ / 10⁹ W Hz⁻¹ (eligible AFGK; 32,666 stars up to L_env), and f < 2.3×10⁻⁵ above ~1.1×10¹⁰ W Hz⁻¹
  (205,981 stars). With the control background subtracted: 4.5×10⁻² / 1.1×10⁻³ / 1.2×10⁻⁴ and 2.0×10⁻⁵.
  At L_env itself f < 4.0×10⁻⁵ (envelope stars detected with probability ≈ 0.5). Without the multiplicity criterion
  (7 detections among 46,206 AFGK stars, μ95 = 13.1): f < 2.8×10⁻⁴ below L_env.
  Restricted to the 20,421 eligible stars whose activity is X-ray-constrained below −4 (04b xray_constrained; TYC 3573-1391-1
  is unconstrained): k = 0, f < 1.47×10⁻⁴ above 10⁹ W Hz⁻¹ (adopted 1.46×10⁻⁴).
  M dwarfs with every coadd detection counted (72 of 159,937, 16 of them secure natural associations; no eligibility decision):
  f < 0.16 / 4.9×10⁻³ / 5.5×10⁻⁴ above 10⁷ / 10⁸ / 10⁹ W Hz⁻¹ (Fig. 7, blue dashed).
  Duty cycle, counted per host (independent snapshots), above L_env: f < 3.8×10⁻⁵ / 5.3×10⁻⁵ / 1.2×10⁻⁴ for δ = 1 / 0.3 / 0.1, f·δ < 1.1×10⁻⁵
  as δ → 0 without background subtraction (2.9×10⁻⁵ / 4.2×10⁻⁵ / 9.7×10⁻⁵ and 8.5×10⁻⁶ background-subtracted).
  Within 50 pc (4,103 eligible stars sensitive to the tabulated 50 pc EIRP, 0 steady detections), common construction
  (P_det = 0.5, no background): f < 1.5×10⁻³, ~4× tighter than the 0.60% for the 1,005 GBT S-band targets of Price et al. (2020).
- **Sample:** 91% of the 229,425 non-WD GCNS objects within 100 pc at δ > −41° (03d); 35 bright dwarfs missing from
  Gaia measured at SIMBAD positions (03c): 3 detected (Algol, θ Aur, ξ Tau), Sirius L_ν < 5.4×10⁵ W Hz⁻¹.
  Activity evidence: 73% of stars within 10 pc, 27% at 20–40 pc, 9% at 80–100 pc.
- **Stacks (all epochs):** M-dwarf mean > 4σ in 4 of 6 shells, 20–40 pc: (8.1×10⁵ ± 9.2×10⁴) W Hz⁻¹ (8.8σ),
  of which the stars with activity evidence give (1.9×10⁶ ± 1.7×10⁵) (11.3σ) and those without (4.3×10⁵ ± 1.1×10⁵)
  (4.1σ); median S/σ shift including detected stars 0.084 ± 0.014. Stack limits use the eligible AFGK stars only (no M-dwarf
  transmitter limits below L_env), a per-L_T host-S/N ≤ 3 selection and host-sampling variance: f < 1 at L_T ≈ 1.4×10⁵ W Hz⁻¹; at 10⁶,
  f < 0.30 with N_eff = 77 (nearest stars); coverage ≥ 0.88 over L_T = 1.5×10⁵–10⁷ W Hz⁻¹ and f = 0.1–1 (8 noise realizations × 25 host
  draws; approximate near f = 1). Median stack (pooled 0–100 pc): f < 0.084 at 10⁷ and 0.016 at 10⁹ W Hz⁻¹.
- **Versus Breakthrough Listen (Price et al. 2020, with the ×3.3 turboSETI correction of Choza et al. 2024; also applied in Fig. 9 to Enriquez et al. 2017, Price et al. 2020, Garrett & Siemion 2023 and Uno et al. 2023):**
  - VLASS (all epochs) is about 43,100× deeper in flux density (0.54 mJy vs ~23 Jy at 50 pc).
  - For narrowband lines it is about 4.6×10⁴× less sensitive in EIRP (1.5×10⁵× against the published 2.1×10¹² W).
  - A search of the same GBT data matched to bandwidth B would lose sensitivity only as √B, so the continuum advantage is coverage
    (signals the turboSETI pipeline does not search for) and every star in the volume, not depth.

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
  - Limits for non-detected stars are (5 + 1.645) σ: in the Epoch-1 catalogue analysis σ is the subtile MeanIslrms; in the
    all-epoch analysis it is the calibrated forced-photometry uncertainty (17) of the coadd.
  - EIRP = L_ν × 2 GHz is the total in-band power of a steady signal of any width that survives RFI flagging and imaging.
- **Duty cycle:** each QL position is observed for only a few seconds. For a transmitter on in a random fraction δ of snapshots,
  the all-epoch analysis counts host stars detected at least once, with probability 1 − ∏_e(1 − δ p_ie) (independent
  snapshots); only as δ → 0 does this reduce to a limit on f × δ. The Epoch-1 single-snapshot limits are on f × δ.
- **Stack interpretation:**
  - Excluding individually detected stars truncates each shell at a luminosity threshold proportional to d². That is why the M-dwarf mean rises with distance.
  - Epoch-1 stacks (10): each shell is used only for transmitter powers below its 5th-percentile per-star limit.
  - All-epoch mean-stack limits (19): for each L_T only stars where the transmitter would have expected coadd S/N ≤ 3 are
    stacked, and the limits are approximate limits for transmitters of a common luminosity L_T.
  - M-dwarf stack limits are conservative because natural emission is present.

## Citation of data

Gaia DR3 (Gaia Collaboration 2023); VLASS (Lacy et al. 2020) and the CIRADA Epoch 1 QL catalogue (Gordon et al. 2021,
ApJS 255, 30); NRAO interim QL and median-stack component lists (Lacy, Dong et al. 2025, VLASS Memo 22) and Single Epoch
component lists (NRAO); ROSAT 2RXS (Boller et al. 2016); eRASS1 (Merloni et al. 2024); SIMBAD (Wenger et al. 2000);
DSS (STScI); Pan-STARRS1 (Chambers et al. 2016; Flewelling et al. 2020); CADC; VizieR (Ochsenbein et al. 2000).
