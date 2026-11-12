"""Build refs.bib from DOIs via the CrossRef BibTeX transform (no hand-typed bibliographic data except the
non-CrossRef entries in MANUAL, which are arXiv/ASCL records). Re-run: python build_bib.py"""
import re, sys, time, requests

import unicodedata
SPECIAL = {"ß": r"{\ss}", "ø": r"{\o}", "Ø": r"{\O}", "ł": r"{\l}", "Ł": r"{\L}", "İ": r"{\.I}", "ı": r"{\i}", "æ": r"{\ae}"}
ACCENT = {"\u0301": "'", "\u0300": "`", "\u0302": "^", "\u0303": "~", "\u0308": '"', "\u030c": "v",
          "\u030b": "H", "\u0307": ".", "\u0327": "c", "\u0328": "k", "\u0306": "u", "\u030a": "r", "\u0304": "="}

def texify(text):
    """Pure-ASCII BibTeX: BibTeX is byte-oriented and splits multi-byte UTF-8 when abbreviating first names."""
    out = []
    for ch in text:
        if ord(ch) < 128: out.append(ch); continue
        if ch in SPECIAL: out.append(SPECIAL[ch]); continue
        d = unicodedata.normalize("NFD", ch)
        if len(d) == 2 and d[1] in ACCENT:
            base = {"i": r"\i", "j": r"\j"}.get(d[0], d[0])
            sep = " " if ACCENT[d[1]].isalpha() else ""
            out.append("{\\" + ACCENT[d[1]] + sep + base + "}"); continue
        raise ValueError(f"no TeX mapping for {ch!r} (U+{ord(ch):04X})")
    return "".join(out)

def finalize(text):
    text = re.sub(r",\s*month\s*=\s*\{?\w+\}?", "", text)   # AAS styles do not print months; CrossRef uses non-standard 'Sept'
    # CrossRef drops corporate authors (leaving "author={ and ...") and keeps HTML / transliterations
    corp = {"Astropy2013": "Astropy Collaboration", "Astropy2018": "Astropy Collaboration", "Astropy2022": "Astropy Collaboration",
            "GaiaMission": "Gaia Collaboration", "GaiaDR3": "Gaia Collaboration", "Smart2021": "Gaia Collaboration"}
    def fix_entry(e):
        m = re.match(r"\s*\w+\{([^,]+),", e)                    # entry text after splitting on "@"
        if m and m.group(1) in corp:
            e = re.sub(r"author=\{\s*(?:\{[^{}]*Collaboration\}|The Astropy Collaboration)?\s*and ",
                       "author={{" + corp[m.group(1)] + "} and ", e, count=1)
        elif "author={ and" in e:
            raise ValueError(f"empty leading author in {m.group(1) if m else e[:40]}")
        return e
    text = "@".join(fix_entry(e) if i else e for i, e in enumerate(text.split("@")))
    JOURNALS = {  # AAS reference style uses the standard abbreviations (AASTeX journal macros where defined)
        "The Astronomical Journal": r"\aj", "The Astrophysical Journal": r"\apj", "The Astrophysical Journal Letters": r"\apjl",
        "The Astrophysical Journal Supplement Series": r"\apjs", r"Astronomy \& Astrophysics": r"\aap",
        "Astronomy and Astrophysics Supplement Series": r"\aaps", "Monthly Notices of the Royal Astronomical Society": r"\mnras",
        "Publications of the Astronomical Society of the Pacific": r"\pasp", "Annual Review of Astronomy and Astrophysics": r"\araa",
        "Nature": r"\nat", "Nature Astronomy": "NatAs", "Nature Methods": "NatMe", "Research Notes of the AAS": "RNAAS",
        r"Computing in Science \& Engineering": "CSE"}
    text = re.sub(r"journal=\{([^}]*)\}", lambda m: "journal={" + JOURNALS.get(m.group(1), m.group(1)) + "}", text)
    text = re.sub(r"\s*<sup>.*?</sup>", "", text)
    text = re.sub(r"(@inproceedings\{McKinney2010,) series=\{SciPy\},", r"\1", text)   # avoid "SciPy (SciPy)"
    text = re.sub(r"(McKinney2010,.*?) publisher=\{SciPy\},", r"\1", text, flags=re.S)
    text = text.replace("Guedel, Manuel", "G\u00fcdel, Manuel")
    # MNRAS now uses article numbers (e.g. stag991) that CrossRef does not put in a pages field
    def add_mnras_eid(e):
        if "journal={\\mnras}" in e and "pages=" not in e:
            d = re.search(r"DOI=\{10\.1093/mnras/(\w+)\}", e)
            if d: e = re.sub(r"(year=\{\d{4}\})", r"\1, pages={" + d.group(1) + "}", e, count=1)
        return e
    text = "@".join(add_mnras_eid(e) if i else e for i, e in enumerate(text.split("@")))
    # CrossRef 'issued' is the online date; use the volume year where they differ
    text = re.sub(r"(Garrett2022,.*?year=\{)2022(\})", r"\g<1>2023\2", text, count=1, flags=re.S)
    return texify(text)

DOIS = {
    "Price2020": "10.3847/1538-3881/ab65f1", "Enriquez2017": "10.3847/1538-4357/aa8d1b",
    "Isaacson2017": "10.1088/1538-3873/aa5800", "Tremblay2025": "10.3847/1538-3881/ad9ea5",
    "Painter2025": "10.3847/1538-3881/adbc5e", "Wlodarczyk2020": "10.1093/mnras/staa2672",
    "Garrett2022": "10.1093/mnras/stac2607",
    "Garrett2026": "10.1093/mnras/stag991", "Saide2023": "10.1093/mnras/stad378", "Wright2018": "10.3847/1538-3881/aae099",
    "Tarter2001": "10.1146/annurev.astro.39.1.511", "Lacy2020": "10.1088/1538-3873/ab63eb",
    "Gordon2021": "10.3847/1538-4365/ac05c0", "Gordon2020": "10.3847/2515-5172/abbe23",
    "GaiaDR3": "10.1051/0004-6361/202243940", "GaiaMission": "10.1051/0004-6361/201629272",
    "Smart2021": "10.1051/0004-6361/202039498", "GudelBenz1993": "10.1086/186766",
    "Vedantham2022": "10.3847/2041-8213/ac5115", "Callingham2021": "10.1038/s41550-021-01483-0",
    "Boller2016": "10.1051/0004-6361/201525648", "Merloni2024": "10.1051/0004-6361/202347165",
    "Wenger2000": "10.1051/aas:2000332", "Ochsenbein2000": "10.1051/aas:2000169",
    "Condon1998": "10.1086/300337", "Becker1995": "10.1086/176166", "Helfand2015": "10.1088/0004-637x/801/1/26",
    "Ivezic2002": "10.1086/344069", "deVries2004": "10.1086/383550", "Flewelling2020": "10.3847/1538-4365/abb82d",
    "Pecaut2013": "10.1088/0067-0049/208/1/9", "Astropy2013": "10.1051/0004-6361/201322068",
    "Astropy2018": "10.3847/1538-3881/aabc4f", "Astropy2022": "10.3847/1538-4357/ac7c74",
    "Harris2020": "10.1038/s41586-020-2649-2", "Virtanen2020": "10.1038/s41592-019-0686-2",
    "Hunter2007": "10.1109/MCSE.2007.55", "McKinney2010": "10.25080/Majora-92bf1922-00a",
}
MANUAL = r"""
@ARTICLE{Chambers2016,
  author = {{Chambers}, K.~C. and {Magnier}, E.~A. and {Metcalfe}, N. and others},
  title = "{The Pan-STARRS1 Surveys}",
  journal = {arXiv e-prints}, year = 2016, eid = {arXiv:1612.05560},
  eprint = {1612.05560}, archivePrefix = {arXiv}, primaryClass = {astro-ph.IM}}
@ARTICLE{Mohan2015,
  author = {{Mohan}, Niruj and {Rafferty}, David},
  title = "{PyBDSF: Python Blob Detection and Source Finder}",
  journal = {Astrophysics Source Code Library, record ascl:1502.007}, year = 2015}
@ARTICLE{Graham2014,
  author = {{Graham}, Matthew and {Plante}, Ray and {Tody}, Doug and {Fitzpatrick}, Mike},
  title = "{PyVO: Python access to the Virtual Observatory}",
  journal = {Astrophysics Source Code Library, record ascl:1402.004}, year = 2014}
"""

if __name__ == "__main__":
    if "--fix-only" in sys.argv:                                 # re-clean an existing refs.bib without network
        txt = open("refs.bib").read()                            # read fully BEFORE opening for write
        open("refs.bib", "w").write(finalize(txt)); print("cleaned refs.bib"); raise SystemExit
    out = []
    for key, doi in DOIS.items():
        for t in range(4):
            r = requests.get(f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex", timeout=60)
            if r.status_code == 200 and r.text.lstrip().startswith("@"): break
            time.sleep(3 * (t + 1))
        else:
            raise SystemExit(f"failed: {key} {doi}")
        bib = re.sub(r"^(\s*@\w+\{)[^,]+,", lambda m: m.group(1) + key + ",", r.text.strip(), count=1)
        bib = re.sub(r"\s*</?i>\s*", " ", bib).replace("&amp;", r"\&")
        bib = re.sub(r"\s+", " ", bib)                                # CrossRef sometimes breaks titles across lines
        bib = bib.replace("\u2013", "--").replace("\u2014", "---").replace("\u2019", "'").replace("Core Package*}", "Core Package}")
        bib = re.sub(r"title=\{([^}]*)\}", lambda m: "title={{" + m.group(1).strip() + "}}", bib, count=1)  # protect capitalisation
        out.append(bib); time.sleep(0.5)
    open("refs.bib", "w").write(finalize("\n\n".join(out) + "\n" + MANUAL))
    print(f"wrote refs.bib with {len(out)} CrossRef + 3 manual entries")
