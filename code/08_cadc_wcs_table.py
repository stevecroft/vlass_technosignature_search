"""08 - Per-image WCS for every VLASS Quick Look science image held at CADC (from caom2.Chunk), so pixel cutout
boxes can be placed without downloading FITS headers. Async TAP job, ~6 min.
Output: data/raw/vlass_ql_wcs_cadc.parquet (122,698 images: epochs 1.1-3.2 complete, 4.1 partial, as of Nov 2026)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import pyvo

QUERY = """SELECT o.observationID, a.uri, c.position_axis_axis1_ctype AS ctype1,
c.position_axis_function_dimension_naxis1 AS nx, c.position_axis_function_dimension_naxis2 AS ny,
c.position_axis_function_refCoord_coord1_pix AS crpix1, c.position_axis_function_refCoord_coord2_pix AS crpix2,
c.position_axis_function_refCoord_coord1_val AS crval1, c.position_axis_function_refCoord_coord2_val AS crval2,
c.position_axis_function_cd11 AS cd11, c.position_axis_function_cd12 AS cd12, c.position_axis_function_cd21 AS cd21, c.position_axis_function_cd22 AS cd22,
p.time_bounds_lower AS mjd
FROM caom2.Observation AS o JOIN caom2.Plane AS p ON o.obsID=p.obsID JOIN caom2.Artifact AS a ON a.planeID=p.planeID
JOIN caom2.Part AS pt ON pt.artifactID=a.artifactID JOIN caom2.Chunk AS c ON c.partID=pt.partID
WHERE o.collection='VLASS' AND p.productID LIKE '%quicklook' AND a.productType='science' AND a.uri LIKE '%tt0.subim.fits'"""

if __name__ == "__main__":
    job = pyvo.dal.TAPService(C.CADC_TAP).submit_job(QUERY, maxrec=500000)
    job.run(); job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=1800)
    wq = job.fetch_result().to_table().to_pandas()
    wq["epoch"] = wq.uri.str.extract(r"VLASS(\d\.\d)\.ql")[0]
    out = os.path.join(C.RAW, "vlass_ql_wcs_cadc.parquet"); wq.to_parquet(out)
    print(len(wq), wq.epoch.value_counts().sort_index().to_dict())
