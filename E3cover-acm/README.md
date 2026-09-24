# E3CoverNet

**Progressive E(3)-aware 3D Backbone for Geometry-Sensitive Vision-Language Alignment**
ACM International Conference on Multimedia (**ACM MM 2026**)

E3CoverNet performs geometry-sensitive vision-language alignment for identifying
objects in real-world 3D scenes from natural-language descriptions. The
repository contains the model, data preparation utilities, training and
evaluation code, and the Nr3D/Sr3D data-processing pipeline.

All commands below assume the current directory is `E3cover-acm/`, the directory
that contains this README and `setup.py`.

## Repository layout

```text
E3cover-acm/
├── e3covernet/
│   ├── analysis/          # Prediction and language-analysis utilities
│   ├── data/              # Metadata, mappings, and dataset split files
│   ├── data_generation/   # Nr3D and Sr3D preparation tools
│   ├── external_tools/    # Vendored Scan2CAD and PointNet++ code
│   ├── in_out/            # Dataset loading and command-line arguments
│   ├── models/            # Listener, geometry backbone, fusion, and task heads
│   ├── scripts/           # Training and preprocessing entry points
│   └── utils/             # Evaluation, logging, and visualization helpers
├── images/                 # Documentation assets
├── tests/                  # Layout and numerical model checks
├── LICENSE
├── README.md
└── setup.py
```

All Python imports use the `e3covernet` namespace. There are two intentionally
distinct `E3CoverNet` classes:

- `e3covernet.models.backbone.e3covernet.E3CoverNet` is the progressive
  geometry backbone used by the new grounding and retrieval heads.
- `e3covernet.models.e3covernet.E3CoverNet` is the original listener model used
  by `train_e3covernet.py`.

## Architecture

The implementation includes the progressive geometry backbone, grounding and
retrieval model components, and the dataset-backed listener training path:

- `models/backbone/e3covernet/`: radial-basis distance encoding, learned
  E(1)/E(2)/E(3) coverings, geometry-aware attention, progressive lift blocks,
  and the three-stage backbone.
- `models/fusion/`: bidirectional cross-attention between language tokens and
  object features.
- `models/grounding/`: the unified 3D visual-grounding network.
- `models/losses/` and `models/retrieval/`: symmetric InfoNCE and the
  text-to-shape dual-tower model.
- `tests/test_equivariance.py`: rotation/reflection equivariance, invariance,
  ablation-construction, grounding, and gradient checks.

The listener remains available in `models/e3covernet.py`. The geometry backbone
is imported explicitly from
`e3covernet.models.backbone.e3covernet`, so the two APIs do not shadow one
another.

## Requirements

- Python 3.8 or newer (as declared by `setup.py`)
- PyTorch; install the CPU or CUDA build appropriate for the local system
- Transformers for the default frozen BERT text encoder
- The Python dependencies declared in [`setup.py`](setup.py)
- A CUDA-capable PyTorch installation, CUDA toolkit, and C++ compiler to build
  the PointNet++ extension required by the original listener

## Installation

Run the following commands from the directory containing this README:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Install the appropriate PyTorch build first; see https://pytorch.org/get-started/locally/
python -m pip install torch
python -m pip install -e .
```

PyTorch is deliberately installed separately because its correct package/index
depends on whether the machine uses CPU, CUDA, or another accelerator. The
editable install provides the remaining declared dependencies, including
Transformers and NLTK.

`pip` may create an isolated build environment and download build tools. If the
machine uses a restricted package proxy, configure that proxy or install
`setuptools` and `wheel` from the organization's package mirror before running
the editable-install command.

To build the PointNet++ CUDA extension used by the original listener:

```bash
cd e3covernet/external_tools/pointnet2
python setup.py install
cd ../../../
```

The progressive backbone itself does not require this extension and can run on
CPU. `train_e3covernet.py` is CUDA-only and defaults to the PointNet++ object
encoder, so dataset-backed listener training needs both a CUDA device and the
extension.

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

The command writes
`/path/to/preprocessed-data/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl`
with the defaults. For Sr3D, add `--process-only-zero-view false` so all scan
views are retained.

### Language data

The listener consumes a processed Nr3D or Sr3D CSV, not the raw download. First
download the NLTK tokenizer data:

```bash
python -m nltk.downloader punkt punkt_tab
```

Prepare Nr3D with a newline-delimited English vocabulary (for example, one
derived from the GloVe vocabulary):

```bash
python e3covernet/scripts/prepare_referential_data.py \
  -scannet-file /path/to/scannet.pkl \
  -type nr3d \
  -out-file /path/to/nr3d-preprocessed.csv \
  --nr3d-file /path/to/nr3d.csv \
  --vocab-file /path/to/glove-vocabulary.txt
```

For Sr3D, no external vocabulary or spell-checking dictionary is needed:

```bash
python e3covernet/scripts/prepare_referential_data.py \
  -scannet-file /path/to/scannet.pkl \
  -type sr3d \
  -out-file /path/to/sr3d-preprocessed.csv \
  --sr3d-file /path/to/sr3d.csv
```

The bundled SymSpell dictionary is selected automatically. It can be replaced
with `--word-freq-file /path/to/dictionary.txt`. Sr3D generation utilities and
configuration files are under
[`e3covernet/data_generation/sr3d`](e3covernet/data_generation/sr3d).

## Listener training

After installing the package, run training from the project root:

```bash
python e3covernet/scripts/train_e3covernet.py \
  -scannet-file /path/to/scannet.pkl \
  -e3covernet-file /path/to/references.csv \
  --log-dir /path/to/logs \
  --n-workers 4
```

Here `-scannet-file` is the pickle produced by ScanNet preprocessing and
`-e3covernet-file` is one of the processed language CSVs above. A CUDA device
and the PointNet++ extension are required. `--log-dir` is mandatory for a new
training run; the parser accepts `--resume-path` instead when resuming.

To augment Nr3D training with Sr3D, add:

```bash
--augment-with-sr3d /path/to/sr3d.csv
```

## Listener evaluation

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

## Equivariance validation

```bash
python -m unittest tests/test_package_layout.py
python tests/test_equivariance.py
```

The first command performs dependency-free package-layout regression checks.
The second requires PyTorch and checks both rotations and reflections, the
complete backbone, symmetric shapes, supported ablation variants, the grounding
pipeline, and finite backward gradients.

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
