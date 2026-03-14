# DETECTION_GENAI

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Research](https://img.shields.io/badge/Research-AI%20Text%20Detection-green)

Repository containing the implementation and experimental results for
the research project on **detecting AI-generated texts using Machine
Learning and Deep Learning techniques**.

The project explores multiple neural architectures including **CNNs,
MobileNet-based models, and Transformer-based models** to distinguish
between **human-written texts and AI-generated texts**.

The experiments are implemented using **Python and Jupyter Notebooks**,
with execution primarily performed in **Google Colab environments**.

------------------------------------------------------------------------

# Research Context

The rapid diffusion of **Generative AI models** (e.g., Large Language
Models) has made the automatic detection of AI-generated content an
important research problem.

This repository investigates the effectiveness of different **neural
architectures for AI text detection**, evaluating their performance
across multilingual datasets.

------------------------------------------------------------------------

# Repository Structure

    DETECTION_GENAI
    │
    ├── Notebook_colab/
    │
    ├── Results/
    │
    ├── Overview_DetectionAI.xlsx
    │
    ├── dt_train_art&mh.csv
    ├── dt_test_art&mh.csv
    └── dt_eval_art&mh.csv

------------------------------------------------------------------------

# Notebook_colab

This directory contains all **Jupyter Notebooks used for model
development, training, and evaluation**.

The notebooks include:

-   data preprocessing
-   model architecture definition
-   training pipelines
-   evaluation metrics
-   final experimental outputs

### CNN-based Models

-   `CNN-DETECTOR_ART&MH.ipynb`
-   `CNN-DETECTOR_dtEN.ipynb`
-   `CNN-DETECTOR_dtITA.ipynb`

These notebooks implement **Convolutional Neural Network classifiers**
for detecting AI-generated texts.

------------------------------------------------------------------------

### MobileNet-based Detector

`CNNmobilenet-DETECTOR_dtEN.ipynb`

Implements a **MobileNet-inspired architecture** optimized for
classification efficiency.

------------------------------------------------------------------------

### Custom Detection Models

-   `DETECTOR_ART&MH.ipynb`
-   `DETECTOR_dtEN.ipynb`
-   `DETECTOR_dtITA.ipynb`

These notebooks implement **custom architectures designed specifically
for AI-text detection tasks**.

------------------------------------------------------------------------

### Transformer-based Model

`TRANSFORMER-DETECTOR_dtEN.ipynb`

Implements a **Transformer-based architecture** for detecting
AI-generated content.

Transformers are particularly effective for modeling long-range
dependencies in text.

------------------------------------------------------------------------

### Utility Notebook

`extractor_for_dt.ipynb`

Used to extract features and prepare datasets for training.

------------------------------------------------------------------------

### Experimental Test Notebooks

The following notebooks perform **controlled evaluation experiments**:

-   `test_singleclass_withCNN-DETECTOR_ART&MH.ipynb`
-   `test_singleclass_withCNNmobilenet-DETECTOR_dtEN.ipynb`
-   `test_singleclass_withDETECTOR_ART&MH.ipynb`
-   `test_singleclass_withDETECTOR_dtEN.ipynb`
-   `test_singleclass_withTRANSFORMER-DETECTOR_dtEN.ipynb`

These experiments help analyze **model robustness and performance under
simplified classification scenarios**.

------------------------------------------------------------------------

# Results

The `Results` folder contains experimental outputs stored as **Excel
files**.

These files report evaluation metrics including:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Training statistics
-   Validation statistics

### Files included

-   `en_train-DetectionAI.xlsx`
-   `en_dev-DetectionAI.xlsx`
-   `it_train-DetectionAI.xlsx`
-   `it_dev-DetectionAI.xlsx`
-   `mydt_train-DetectionAI.xlsx`
-   `mydt_test-DetectionAI.xlsx`

------------------------------------------------------------------------

# Dataset

Datasets used for the experiments are provided in **CSV format**.

### Files

-   `dt_train_art&mh.csv`
-   `dt_test_art&mh.csv`
-   `dt_eval_art&mh.csv`

These datasets contain labeled examples of:

-   **Human-written text**
-   **AI-generated text**

used to train and evaluate the detection models.

------------------------------------------------------------------------

# Results Overview

`Overview_DetectionAI.xlsx`

This file aggregates the main experimental results across all evaluated
models.

It allows quick comparison of model performance.

------------------------------------------------------------------------

# Technologies

The project relies on the following technologies:

-   Python
-   Jupyter Notebook
-   Google Colab
-   Deep Learning frameworks (TensorFlow / PyTorch)
-   Scikit-learn

------------------------------------------------------------------------

# Reproducibility

To reproduce the experiments:

1.  Clone the repository

```{=html}
<!-- -->
```
    git clone https://github.com/USERNAME/DETECTION_GENAI.git

2.  Open the notebooks in **Google Colab** or **Jupyter Notebook**

3.  Install required libraries

```{=html}
<!-- -->
```
    pip install numpy pandas scikit-learn tensorflow torch

4.  Run the notebooks sequentially.

------------------------------------------------------------------------

# Citation

If you use this repository in your research, please cite the
corresponding work.

    @misc{buttaro2026genai_detection,
      author = {Cristian Buttaro},
      title = {Detection of AI-Generated Texts Using Neural Architectures},
      year = {2026},
      note = {GitHub repository}
    }

------------------------------------------------------------------------

# Author

Cristian Buttaro

------------------------------------------------------------------------

# License

This project is released for **research purposes**.
