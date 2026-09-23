# E3CoverNet

**Progressive E(3)-aware 3D Backbone for Geometry-Sensitive Vision-Language Alignment**
ACM International Conference on Multimedia (**ACM MM 2026**)

E3CoverNet performs geometry-sensitive vision-language alignment for identifying
objects in real-world 3D scenes from natural-language descriptions. The
repository contains the model, data preparation utilities, training and
evaluation code, and the Nr3D/Sr3D data-processing pipeline.

## Repository layout

```text
E3cover-acm/
├── e3covernet/
│   ├── analysis/          # Prediction and language-analysis utilities
│   ├── data/              # Metadata, mappings, and dataset split files
│   ├── data_generation/   # Nr3D and Sr3D preparation tools
│   ├── in_out/            # Dataset loading and command-line arguments
│   ├── models/            # E3CoverNet and its backbone modules
│   ├── scripts/           # Training and preprocessing entry points
│   └── utils/             # Evaluation, logging, and visualization helpers
├── LICENSE
├── README.md
└── setup.py
```

All Python imports use the `e3covernet` namespace. The main model class is
`e3covernet.models.e3covernet.E3CoverNet`.

## Requirements

- Python 3
- PyTorch with a CUDA version compatible with the local system
- The Python dependencies declared in [`setup.py`](setup.py)
- A C++/CUDA build toolchain when using the PointNet++ extension

## Installation

Run the following commands from the directory containing this README:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

`pip` may create an isolated build environment and download build tools. If the
machine uses a restricted package proxy, configure that proxy or install
`setuptools` and `wheel` from the organization's package mirror before running
the editable-install command.

To build the optional PointNet++ CUDA extension:

```bash
cd e3covernet/external_tools/pointnet2
python setup.py install
cd ../../../
```

## Data preparation

### ScanNet

Download ScanNet and follow the preprocessing instructions in
[`e3covernet/data/scannet/README.md`](e3covernet/data/scannet/README.md). Then
prepare the scan pickle from the project root:

```bash
python e3covernet/scripts/prepare_scannet_data.py \
  -top-scan-dir /path/to/scannet/scans \
  -top-save-dir /path/to/preprocessed-data
```

Use `python e3covernet/scripts/prepare_scannet_data.py --help` to confirm the
arguments supported by the local checkout.

### Language data

The training pipeline accepts prepared Nr3D or Sr3D CSV files. Sr3D generation
utilities and configuration files are under
[`e3covernet/data_generation/sr3d`](e3covernet/data_generation/sr3d).

## Training

After installing the package, run training from the project root:

```bash
python e3covernet/scripts/train_e3covernet.py \
  -scannet-file /path/to/scannet.pkl \
  -e3covernet-file /path/to/references.csv \
  --log-dir /path/to/logs \
  --n-workers 4
```

To augment Nr3D training with Sr3D, add:

```bash
--augment-with-sr3d /path/to/sr3d.csv
```

## Evaluation

```bash
python e3covernet/scripts/train_e3covernet.py \
  --mode evaluate \
  -scannet-file /path/to/scannet.pkl \
  -e3covernet-file /path/to/references.csv \
  --resume-path /path/to/checkpoints/best_model.pth \
  --n-workers 4 \
  --batch-size 64
```

Run `python e3covernet/scripts/train_e3covernet.py --help` for the complete set
of model, dataset, optimization, and logging options.

## Citation

```bibtex
@inproceedings{e3covernet2026,
  title     = {E3CoverNet: Progressive E(3)-aware 3D Backbone for Geometry-Sensitive Vision-Language Alignment},
  booktitle = {ACM International Conference on Multimedia (ACM MM)},
  year      = {2026}
}
```

## License

This project is distributed under the [MIT License](LICENSE).
