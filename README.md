# Reproduction Package

This directory contains the core code and trained checkpoints needed to reproduce training, test-time inverse-design evaluation, benchmark comparison, visualization analysis, and the MATLAB-based structural sensitivity analysis.

## Recommended environment

- Python 3.10
- MATLAB R2022b or newer
- PyTorch 2.2.2
- CUDA-enabled GPU is recommended for training and large-batch evaluation

## Directory layout

- `python/`
  - `bicbam_network.py`: network definitions.
  - `data_utils.py`: Excel parsing, dataset construction, and split helpers.
  - `metrics_utils.py`: SSIM/PCC and decoding utilities.
  - `train_model.py`: training entry point.
  - `evaluate_model.py`: test-time evaluation entry point.
- `benchmark/`
  - `benchmark_inverse_search.py`: traditional-method comparison.
  - `benchmark_utils.py`: benchmark utilities.
  - `benchmark_report.py`: benchmark summary tables and plots.
- `visualization/`
  - `attention_occlusion.py`: occlusion-based analysis.
  - `attention_representative.py`: representative attention-map visualization.
- `matlab/`
  - `sensitivity_analysis_from_excel.m`
  - `create_sensitivity_figures.m`
- `models/`
  - trained PyTorch checkpoints. Model download link [Baidu Drive](https://pan.baidu.com/s/1QPyTK8_lsDlnafXVmmAeJA) After downloading, store it in this folder.
- `docs/`
  - upload and curation notes.
- `requirements.txt`

## Install dependencies

```bash
pip install -r requirements.txt
```

## Python training

```bash
cd python
python train_model.py ^
  --data-root <path-to-image-folder> ^
  --excel-path <path-to-label-excel> ^
  --output-dir ../outputs/train_repro
```

## Python evaluation

```bash
cd python
python evaluate_model.py ^
  --data-root <path-to-image-folder> ^
  --excel-path <path-to-label-excel> ^
  --model-dir ../models ^
  --output-dir ../outputs/test_repro ^
  --test-sample-size 500
```

## Benchmark comparison

```bash
cd benchmark
python benchmark_inverse_search.py --data-root <path-to-image-folder> --excel-path <path-to-label-excel> --model-dir ../models
python benchmark_report.py --result-dir ./benchmark_results
```

## Visualization analysis

The visualization scripts use configurable path variables near the top of each script. Set the image folder, Excel file, model directory, and output directory before running.

## MATLAB scripts

Run in MATLAB:

```matlab
sensitivity_analysis_from_excel
create_sensitivity_figures
```

## Checkpoints

The `models/` folder includes the trained inverse and forward checkpoints needed for direct test reproduction. If you upload this package to GitHub, use Git LFS for `.pth` files.

```bash
git lfs track "*.pth"
```
