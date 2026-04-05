GeoTIFF samples for Streamlit / talks

1) sentinel1_rtc_sample.tif  (~3–4 MB, 1024×1024)
   Real Sentinel-1 IW GRD RTC VV chip (EPSG:32632), georeferenced.
   Regenerate:
     python scripts/download_real_sample_geotiff.py
   Options: --chip 512 --out path/to/out.tif

2) presentation_sample.tif  (tiny fallback)
   From SAMPLE PNG or synthetic speckle:
     python scripts/build_sample_geotiff.py

Attribution for (1): ATTRIBUTION.txt
