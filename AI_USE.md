# AI use disclosure

This file describes how I used AI assistance in this project, in line with the DSA 8401 course policy.


## My work and decisions

- **Problem and data.** I chose the problem (benign vs. malignant classification of breast
  histopathology images) and the dataset (BreakHis, 100X magnification).
- **Model selection.** I carried out the comparative study of ten models, from classical ML on
  handcrafted features to CNNs and vision transformers
  (`notebooks/breast_cancer_classification_academic_markdown.ipynb`), and chose Swin Tiny from it.
  I wrote this notebook myself and used AI only to debug errors and explain code.
- **Evaluation design.** I found that the random image split leaked patients between training and test,
  and decided to evaluate by patient, with stratified, patient-grouped cross-validation.
- **Decision policy.** I set the threshold rule (malignant recall ≥ 0.95), because a missed cancer costs
  more than a false alarm.
- **System design.** I decided to serve the model through a FastAPI/Docker service with Grad-CAM
  explanations, and to also publish a free static version that runs in the browser.
- **Training and results.** I ran all training on Google Colab, checked the training logs, and chose
  which results and limitations to report.

## How I used the AI assistant

I specified what each component had to do. The assistant wrote most of the implementation code, which I
reviewed, tested and ran.

| Area | AI assistance |
|---|---|
| Comparative study notebook | Debugging and explaining code only |
| Training and tuning notebooks | Code for the cross-validation, threshold selection, baseline and artifact export I specified |
| Scripted pipeline, API service and tests | Converting the notebook code into the `pipeline/` scripts; the FastAPI service, input validation and test suite |
| Web app, deployment and CI | Frontend code, ONNX export, Docker, deployment scripts and CI configuration |
| README and slides | First drafts, which I then edited |

## How I checked the AI-generated work

- I ran the 29 automated tests (`python -m pytest tests`), which cover the API, the input handling, and
  the pipeline's split and threshold logic.
- I checked that the scripted splits reproduce my notebook's patient folds exactly.
- I ran parity checks showing that the browser version gives the same predictions as PyTorch.
- I ran the pipeline end to end and served its exported model through the API.
- I tested the live deployments in a browser at desktop and phone sizes.

## Responsibility

I understand all the submitted code and results, and I can explain and defend the methods and
decisions in this project. Any errors are my responsibility.

Catherine Maina · November 2026
