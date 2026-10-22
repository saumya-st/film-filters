# film-filters

**Current presets (`dclassic`, `cpm35`, plus `_soft` = 60% colour cast) are estimates built by `design_looks.py`
from anchor colours in the sample photos in `Desktop/filter/` (no iPhone available).** The chart workflow below
gives exact measurements if anyone with Dazz can run the charts later.

Measure a film-camera filter (e.g. Dazz Cam's D Classic / CPM35) from exports of test charts and rebuild it as
our own `.cube` LUT + grain / bloom / vignette / softness parameters. Nothing is extracted from the app itself.

## 1. Get the charts onto the iPhone
`charts/` has 6 PNGs (3024x4032): `chart_color_A`, `chart_color_B`, `chart_edge`, `flat_grey`, `flat_black`, `flat_white`.
Send them **uncompressed**: Google Drive / iCloud Drive / email as files, or AirDrop. **Not WhatsApp** (it recompresses).
Save them to Photos.

## 2. Run them through Dazz (once per filter)
1. In Dazz pick the filter (D Classic), tap the import/gallery button, choose a chart, save the result.
2. Date stamp **off**, no frame, default settings. Do all 6 charts.
3. Repeat for CPM35.
4. Also run 2 normal photos (one daylight, one night with flash) through each filter and keep the originals.
5. Send everything back uncompressed into:
   - `Desktop/filter/dclassic/` — the 6 chart exports
   - `Desktop/filter/cpm35/` — the 6 chart exports
   - `Desktop/filter/validation/` — originals + their D Classic / CPM35 versions (any names, tell me which is which)

File names don't matter; charts are recognised automatically. HEIC needs `pip install pillow-heif`.

## 3. Build the presets
```
python analyze.py "C:/Users/saumy/Desktop/filter/dclassic" dclassic
python analyze.py "C:/Users/saumy/Desktop/filter/cpm35" cpm35
```
Writes `presets/<name>.cube`, `<name>.json`, `<name>_report.txt`.

## 4. Use / check them
```
python apply_filter.py photo.jpg dclassic out.jpg [--no-grain] [--date "'26 10 22"]
python compare.py original.jpg dazz_version.jpg dclassic     # dE2000 + side-by-side in report/
```

## Pipeline (same order the Android GLSL shader will use)
input blur -> 3D LUT (trilinear) -> highlight bloom -> vignette -> luminance-dependent grain [-> date stamp]

## Self-test
`python synth_test.py` fakes a "Dazz export" with a known look (incl. resize, 1.2° rotation, JPEG) and checks
recovery. Current result: LUT mean dE2000 0.32 (95% 0.6), blur/vignette/bloom within ~7%, end-to-end photo dE 0.67.
Note: grain colour correlation reads higher than the truth (0.95 vs 0.7) because JPEG chroma subsampling
smooths colour noise — this matches what the real exports look like, so it's left as measured.
