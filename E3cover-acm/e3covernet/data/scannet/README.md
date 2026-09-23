# ScanNet

To get access to the ScanNet scans please refer to [ScanNet Official Repo](https://github.com/ScanNet/ScanNet#scannet-data) for getting the download instructions.
you will need to download the following files for each scan.
```
*.aggregation.json
*.txt, 
*_vh_clean_2.0.010000.segs.json
*_vh_clean_2.ply
*_vh_clean_2.labels.ply
```

# Preprocess data for E3CoverNet

Run preprocessing from the project root. For Nr3D, the default configuration
processes only ScanNet `_00` scenes:

```bash
python e3covernet/scripts/prepare_scannet_data.py \
  -top-scan-dir /path/to/scannet/scans \
  -top-save-dir /path/to/preprocessed-data
```

For Sr3D, process all views by appending
`--process-only-zero-view false`. The script loads each scan's annotated point
clouds and writes the resulting pickle under the requested output directory.
The `scene0009_00` warning can be safely ignored.
