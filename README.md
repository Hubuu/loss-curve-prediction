# Learning-Rate-Schedule-Aware Loss Curve Prediction

This repository contains the code for the final project of **Topics in Deep Learning Theory**.

## Project Goal

The goal is to predict the full pretraining loss curve under a new learning-rate schedule.

Following the course requirement, we fit analytic loss prediction models on the **cosine** learning-rate schedule and evaluate their prediction performance on the **WSD** schedule.

## Methods

The notebook includes the following methods and analyses:

* One-Power Law baseline
* Tissue-style LR-annealing law
* Multi-Power Law
* Diagnostic fitting with full 8-1-1 and 8-1-1 constant segment
* Decoupled fitting
* Ridge residual correction with schedule-derived features
* Residual and stable-phase drift analysis

## Repository Structure

```text
project/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── loss_curve_prediction_final.ipynb
│   ├── src/
│   │   ├── data.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── plots.py
│   │   └── __init__.py
│   └── results/
│       ├── figures/
│       └── tables/
└── loss curves/
    ├── gpt_loss+lrs.pkl
    └── Readme.txt
```

## Data

The loss curve data file is provided by the course.

Please place the data file at:

```text
loss curves/gpt_loss+lrs.pkl
```

The notebook assumes this relative path when loading the dataset.

## Environment

The code was developed with Python 3.11.

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you want to use GPU acceleration for the Multi-Power Law fitting, please install a CUDA-compatible PyTorch version following the official PyTorch installation instructions.

## Reproducing the Results

Open and run the main notebook:

```text
code/loss_curve_prediction_final.ipynb
```

The notebook performs:

1. Data loading and preprocessing
2. Missing-step interpolation and loss smoothing
3. Baseline fitting on cosine and evaluation on WSD
4. Diagnostic experiments with 8-1-1 schedules
5. Decoupled fitting experiments
6. Ridge residual correction
7. Figure and table generation

## Outputs

Generated figures are saved to:

```text
code/results/figures/
```

Generated tables and prediction files are saved to:

```text
code/results/tables/
```

These outputs include the figures and metrics used in the final slides.

## Notes

The `__pycache__` folders and temporary notebook checkpoint files are not needed for reproduction and should not be uploaded.

## Author

Zhirun Han
