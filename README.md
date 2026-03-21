# DETECTION_GENAI

[![arXiv](https://img.shields.io/badge/arXiv-2603.18750-b31b1b.svg)](https://arxiv.org/abs/2603.18750)
[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/cristian03/ARTandMH)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-blue)
![Research](https://img.shields.io/badge/Research-AI%20Text%20Detection-green)

## 📄 Paper

This repository accompanies the paper:<br>
**Automatic Detection of Gen-AI Texts: A Comparative Framework of Neural Models**  
[https://arxiv.org/abs/2603.18750](https://arxiv.org/abs/2603.18750) <br>
The repository provides all the resources required to reproduce the experiments presented in the paper, including datasets, models, and evaluation results.<br>

------------------------------------------------------------------------
This repository contains the implementation and experimental results for the research project on **detecting AI-generated texts using Machine Learning and Deep Learning techniques**.<br>
The project explores multiple neural architectures including **CNNs, MobileNet-based models, and Transformer-based models** to distinguish between **human-written texts and AI-generated texts**.<br>
The experiments are implemented using **Python and Jupyter Notebooks**, with execution primarily performed in **Google Colab environments**.

------------------------------------------------------------------------

# Research Context

The rapid diffusion of **Generative AI models** (e.g., Large Language Models) has made the automatic detection of AI-generated content an
important research problem.<br>
This repository investigates the effectiveness of different **neural architectures for AI text detection**, evaluating their performance across multilingual datasets.

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

This directory contains all **Notebooks used for model development, training, and evaluation**.

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

These notebooks implement **Convolutional Neural Network classifiers** for detecting AI-generated texts.

------------------------------------------------------------------------

### MobileNet-based Detector

`CNNmobilenet-DETECTOR_dtEN.ipynb`

Implements a **MobileNet-inspired architecture** optimized for classification efficiency.

------------------------------------------------------------------------

### MLP-based Detection Models

- `DETECTOR_ART&MH.ipynb`
- `DETECTOR_dtEN.ipynb`
- `DETECTOR_dtITA.ipynb`

These notebooks implement **Multi-Layer Perceptron (MLP) models** for AI-generated text detection.

------------------------------------------------------------------------

### Transformer-based Model

`TRANSFORMER-DETECTOR_dtEN.ipynb`

Implements a **Transformer-based architecture** for detecting AI-generated content.<br>
Transformers are particularly effective for modeling long-range dependencies in text.

------------------------------------------------------------------------

### Utility Notebook : `extractor_for_dt.ipynb`

Used to extract features and prepare datasets for training.

------------------------------------------------------------------------

### Experimental Test Notebooks

The following notebooks perform **evaluation experiments on a single-class Italian dataset** in order to assess the robustness of the previously implemented detection models.

- `test_singleclass_withCNN-DETECTOR_ART&MH.ipynb`
- `test_singleclass_withCNNmobilenet-DETECTOR_dtEN.ipynb`
- `test_singleclass_withDETECTOR_ART&MH.ipynb`
- `test_singleclass_withDETECTOR_dtEN.ipynb`
- `test_singleclass_withTRANSFORMER-DETECTOR_dtEN.ipynb`

These notebooks apply the previously described detectors (CNN-based, MobileNet-based, MLP-based, and Transformer-based models) to a **monoclass Italian dataset**.<br> 
The goal of these experiments is to evaluate the **effective robustness and generalization capability** of the detectors when exposed to a simplified and controlled testing scenario.

------------------------------------------------------------------------

# Results

The `Results` folder contains experimental outputs stored as **Excel files**.<br>
These files report the results obtained from **online AI-text detection tools** used as a benchmark for comparison with the detectors implemented in this repository.<br>
In particular, the Excel files collect the **percentages of AI-generated content detected within the texts of each dataset**. These values allow a direct comparison between the performance of external detectors and the models implemented in this project.

### Files included

- `en_train-DetectionAI.xlsx`
- `en_dev-DetectionAI.xlsx`
- `it_train-DetectionAI.xlsx`
- `it_dev-DetectionAI.xlsx`
- `mydt_train-DetectionAI.xlsx`
- `mydt_test-DetectionAI.xlsx`

Each file contains the detection results for the corresponding dataset, reporting the **percentage of AI detected in the texts analyzed by different online detection systems**.

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

This file aggregates the main experimental results across all evaluated models.<br>
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

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/DETECTION_GENAI.git
cd DETECTION_GENAI
```

### 2. Open the notebooks

Open the notebooks using either:
	•	**Google Colab**
	•	**Jupyter Notebook**

### 3.  Install required libraries

Install the required Python dependecies:
```bash
pip install pandas numpy matplotlib seaborn tqdm torch scikit-learn
```
The notebooks rely on the following Python libraries: **Pandas**, **Numpy**, **Matplotlib.pyplot**, **Seaborn**, **Tqdm.notebook**, **Torch**, **Torch.nn**, **Torch.utils.data**, **Torch.cuda.amp** and **scikit-learn** (`model_selection`,`metrics`).<br><br>
Additionally, several Python standard library modules are used: **os**, **math**, **copy**, **re**, **random**, **collections**. <br> These modules are included in the standard Python distribution and do not require installation.

### 4.  Run the notebooks sequentially.

------------------------------------------------------------------------

# Citation

If you use this repository in your research, please cite the corresponding work.

    @misc{buttaro2026genai_detection,
      author = {Cristian Buttaro},
      title = {Automatic Detection of Gen-AI Texts: A Comparative Framework of Neural Models},
      year = {2026},
      note = {GitHub repository}
    }

------------------------------------------------------------------------

# Author

Cristian Buttaro

------------------------------------------------------------------------

# License

This project is released for **research purposes**.
