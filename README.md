# DETECTION_GENAI

This repository contains the code, notebooks, and experimental results
related to a research project on **detecting AI-generated texts
(Generative AI)** using Machine Learning and Deep Learning techniques.

The project explores different classification models (CNN, MobileNet,
Transformer-based architectures, and custom detectors) to distinguish
between **human-written texts and AI-generated texts** across
multilingual datasets.

All the code is implemented in **Python** using **Jupyter Notebooks**.
Each notebook contains both the implementation and the **final
experimental outputs** produced during model training and evaluation.

------------------------------------------------------------------------

# Repository Structure

    DETECTION_GENAI
    │
    ├── Notebook_colab/
    │   ├── CNN-DETECTOR_ART&MH.ipynb
    │   ├── CNN-DETECTOR_dtEN.ipynb
    │   ├── CNN-DETECTOR_dtITA.ipynb
    │   ├── CNNmobilenet-DETECTOR_dtEN.ipynb
    │   ├── DETECTOR_ART&MH.ipynb
    │   ├── DETECTOR_dtEN.ipynb
    │   ├── DETECTOR_dtITA.ipynb
    │   ├── TRANSFORMER-DETECTOR_dtEN.ipynb
    │   ├── extractor_for_dt.ipynb
    │   ├── test_singleclass_withCNN-DETECTOR_ART&MH.ipynb
    │   ├── test_singleclass_withCNNmobilenet-DETECTOR_dtEN.ipynb
    │   ├── test_singleclass_withDETECTOR_ART&MH.ipynb
    │   ├── test_singleclass_withDETECTOR_dtEN.ipynb
    │   └── test_singleclass_withTRANSFORMER-DETECTOR_dtEN.ipynb
    │
    ├── Results/
    │   ├── en_dev-DetectionAI.xlsx
    │   ├── en_train-DetectionAI.xlsx
    │   ├── it_dev-DetectionAI.xlsx
    │   ├── it_train-DetectionAI.xlsx
    │   ├── mydt_test-DetectionAI.xlsx
    │   └── mydt_train-DetectionAI.xlsx
    │
    ├── Overview_DetectionAI.xlsx
    ├── dt_eval_art&mh.csv
    ├── dt_test_art&mh.csv
    └── dt_train_art&mh.csv

------------------------------------------------------------------------

# Notebook_colab

This folder contains the **Jupyter Notebooks developed and executed in
Google Colab** for implementing and evaluating the detection models.

Each notebook includes:

-   data preprocessing
-   model training
-   evaluation procedures
-   final performance outputs

------------------------------------------------------------------------

## CNN-based Detectors

-   **CNN-DETECTOR_ART&MH.ipynb**\
    Convolutional Neural Network model applied to the ART&MH dataset.

-   **CNN-DETECTOR_dtEN.ipynb**\
    CNN-based detector trained on the English dataset.

-   **CNN-DETECTOR_dtITA.ipynb**\
    CNN-based detector trained on the Italian dataset.

------------------------------------------------------------------------

## MobileNet-based Detector

-   **CNNmobilenet-DETECTOR_dtEN.ipynb**\
    Detector based on the **MobileNet architecture** for AI-generated
    text detection.

------------------------------------------------------------------------

## Custom Detector Models

-   **DETECTOR_ART&MH.ipynb**\
    Custom detection model trained on the ART&MH dataset.

-   **DETECTOR_dtEN.ipynb**\
    Custom detection model trained on the English dataset.

-   **DETECTOR_dtITA.ipynb**\
    Custom detection model trained on the Italian dataset.

------------------------------------------------------------------------

## Transformer-based Detector

-   **TRANSFORMER-DETECTOR_dtEN.ipynb**\
    Transformer-based architecture for detecting AI-generated texts in
    English datasets.

------------------------------------------------------------------------

## Utility Notebook

-   **extractor_for_dt.ipynb**\
    Notebook used to extract and preprocess features from the datasets.

------------------------------------------------------------------------

## Testing Notebooks

These notebooks are used to perform **single-class or controlled testing
scenarios** for evaluating model robustness.

-   **test_singleclass_withCNN-DETECTOR_ART&MH.ipynb**
-   **test_singleclass_withCNNmobilenet-DETECTOR_dtEN.ipynb**
-   **test_singleclass_withDETECTOR_ART&MH.ipynb**
-   **test_singleclass_withDETECTOR_dtEN.ipynb**
-   **test_singleclass_withTRANSFORMER-DETECTOR_dtEN.ipynb**

------------------------------------------------------------------------

# Results

The **Results** folder contains experimental outputs stored in **Excel
format**.

These files include performance metrics such as:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   training and validation statistics

Files included:

-   **en_train-DetectionAI.xlsx** -- training results for the English
    dataset\
-   **en_dev-DetectionAI.xlsx** -- validation results for the English
    dataset\
-   **it_train-DetectionAI.xlsx** -- training results for the Italian
    dataset\
-   **it_dev-DetectionAI.xlsx** -- validation results for the Italian
    dataset\
-   **mydt_train-DetectionAI.xlsx** -- training results for a custom
    dataset\
-   **mydt_test-DetectionAI.xlsx** -- test results for a custom dataset

------------------------------------------------------------------------

# Dataset Files

The repository includes datasets used for training and evaluation in
**CSV format**.

-   **dt_train_art&mh.csv** -- training dataset\
-   **dt_test_art&mh.csv** -- testing dataset\
-   **dt_eval_art&mh.csv** -- evaluation dataset

These datasets contain examples of **human-written and AI-generated
texts** used to train and test the detection models.

------------------------------------------------------------------------

# Results Overview

-   **Overview_DetectionAI.xlsx**

This file provides an aggregated overview of the experimental results
across the different models tested in the project.

------------------------------------------------------------------------

# Technologies Used

The project is implemented using:

-   **Python**
-   **Jupyter Notebooks**
-   **Google Colab**
-   Machine Learning and Deep Learning libraries (TensorFlow / PyTorch /
    Scikit-learn)

------------------------------------------------------------------------

# Project Goal

The goal of this project is to evaluate the effectiveness of different
Machine Learning and Deep Learning architectures for **automatic
detection of AI-generated texts**, comparing multiple models and
multilingual datasets.

------------------------------------------------------------------------

# Author

Developed by:

**Cristian Buttaro**
