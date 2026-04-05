presentation_sample.tif — small single-band GeoTIFF (EPSG:4326) for Streamlit demos.

Regenerate from repo root:
  python scripts/build_sample_geotiff.py

Uses the first noisy SAMPLE patch PNG when data/sample_sar/.../noisy exists;
otherwise writes a synthetic speckle patch. Requires rasterio.
