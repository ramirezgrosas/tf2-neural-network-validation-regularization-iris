# Model Validation and Regularisation on the Iris Dataset

Building, validating and regularising a fully connected neural network in TensorFlow 2 / Keras.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

## Overview

This project trains a neural network that classifies the three species of the Iris dataset, and
walks through the full workflow that turns a model which *memorises* the training set into one that
actually *generalises*.

An intentionally oversized network (79,107 parameters for 114 training samples) is used as the
baseline, so that overfitting is obvious in the learning curves, and is then brought under control
with regularisation and training callbacks.

Everything lives in a single notebook:
[`tf2-neural-network-validation-regularization-iris.ipynb`](tf2-neural-network-validation-regularization-iris.ipynb)

## Dataset

<table>
  <tr>
    <td><img src="data/iris_setosa.jpg" alt="Iris setosa" height="200"/></td>
    <td><img src="data/iris_versicolor.jpg" alt="Iris versicolor" height="200"/></td>
    <td><img src="data/iris_virginica.jpg" alt="Iris virginica" height="200"/></td>
  </tr>
  <tr>
    <td align="center"><b>Iris setosa</b></td>
    <td align="center"><b>Iris versicolor</b></td>
    <td align="center"><b>Iris virginica</b></td>
  </tr>
</table>

The classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris), loaded directly from
`sklearn.datasets`:

|  |  |
|---|---|
| Samples | 150 (50 per class) |
| Features | sepal length, sepal width, petal length, petal width (cm) |
| Classes | *Iris setosa*, *Iris versicolor*, *Iris virginica* |
| Task | multi-class classification |
| Split | 135 train / 15 test, and 15% of the training set reserved for validation (114 train / 21 validation) |

## Workflow

1. **Load and preprocess** — train/test split with scikit-learn and one-hot encoding of the labels.
2. **Build the baseline** — a 10-layer `Sequential` model (64 → 128×4 → 64×4 → 3 softmax) with ReLU
   activations and He uniform initialisation.
3. **Compile and train** — Adam (`lr=0.0001`), categorical cross-entropy, 800 epochs, batch size 40,
   `validation_split=0.15`.
4. **Diagnose overfitting** — accuracy and loss learning curves for training vs. validation.
5. **Regularise** — L2 weight decay, dropout and batch normalisation on the same architecture.
6. **Add callbacks** — `EarlyStopping` on the validation loss and `ReduceLROnPlateau` on the
   learning rate.
7. **Evaluate** — final unbiased measurement on the held-out test set.

## Results

Three training runs of the same architecture, changing only the regularisation and the callbacks:

| Run | Regularisation | Epochs run | Final train acc. | Final val. acc. | Best val. loss | Final val. loss | Gap (val − train loss) |
|-----|----------------|------------|------------------|-----------------|----------------|-----------------|------------------------|
| Baseline | none | 800 | 1.0000 | 0.9048 | 0.1937 (ep. 89) | 0.5358 | 0.534 |
| Regularised | L2 `1e-3` + dropout `0.3` + batch norm | 800 | 1.0000 | 0.9524 | 0.8018 (ep. 785) | 0.8146 | 0.362 |
| Regularised + callbacks | L2 `1e-4` + dropout `0.3` + batch norm | **154** (stopped early) | 0.9561 | 0.9048 | 0.2371 (ep. 124) | 0.2735 | **0.108** |

> The loss of the regularised runs includes the L2 penalty term, so the absolute values are not
> directly comparable between rows. The gap in the last column is: the penalty is added to both the
> training and the validation loss, so it cancels out in the difference.

Final model (regularised + callbacks) on the 15 held-out test samples:

| Metric | Value |
|--------|-------|
| Test loss | **0.209** |
| Test accuracy | **86.67%** (13/15) |

## Key takeaways

- **A perfect training accuracy means nothing on its own.** The baseline drives its training loss
  down to `0.0017` while its validation loss almost triples, from `0.1937` at epoch 89 up to
  `0.5358` at epoch 800: the textbook signature of overfitting.
- **Watch the loss, not only the accuracy.** The baseline's validation accuracy sits around 90-95%
  from epoch 25 to epoch 800 and never signals the problem; the validation loss does. With 21
  validation samples, one sample is worth 4.8% of accuracy, so accuracy alone is too coarse to
  compare these models.
- **Regularisation does not lower the loss, it changes its shape.** L2 + dropout + batch
  normalisation cannot reduce the absolute validation loss (they add a penalty to it), but they
  stop it from diverging: it keeps improving until epoch 785 of 800, the generalisation gap drops
  from `0.534` to `0.362` and the validation accuracy rises from 90.48% to 95.24%. The cost is
  convergence speed, about six times slower to reach the same training accuracy.
- **Callbacks gave the best trade-off:** early stopping ended training at epoch 154 out of 800
  (81% fewer epochs) with the smallest generalisation gap of the three runs, `0.108`, while
  `ReduceLROnPlateau` refined the last epochs by dropping the learning rate from `1e-4` to `2e-5`.
- **Capacity should match the problem.** ~79,000 parameters for 114 samples is far more model than
  Iris needs; it was kept on purpose as the setting where these techniques can be seen at work.

## Project structure

```
.
├── data/                                                    # Images of the three Iris species
├── src/tf2_neural_network_validation_regularization_iris/   # Package scaffold
├── tf2-neural-network-validation-regularization-iris.ipynb  # Main notebook
├── pyproject.toml                                           # Dependencies (uv project)
└── uv.lock                                                  # Pinned versions
```

## Getting started

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python 3.11+.

```bash
# Clone the repository
git clone https://github.com/ramirezgrosas/tf2-neural-network-validation-regularization-iris.git
cd tf2-neural-network-validation-regularization-iris

# Create the virtual environment and install the dependencies
uv sync
```

Then open the notebook in VS Code and select `.venv` as the kernel, or launch Jupyter directly:

```bash
uv run --with jupyter jupyter lab
```

Main dependencies: TensorFlow 2.21 (Keras 3.15), scikit-learn 1.9, NumPy, matplotlib and pandas.

## Notes

Results were produced with `test_size=0.1` and no fixed `random_state` in the train/test split, so
re-running the notebook will not reproduce exactly the same numbers. The conclusions about
overfitting and regularisation hold across runs; the individual figures vary, especially on a test
set of 15 samples.

## Author

**Diego Ramírez Rosas** — [ramirezgrosas](https://github.com/ramirezgrosas)
