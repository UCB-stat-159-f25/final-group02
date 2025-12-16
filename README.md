---
title: Overview
---

# Predicting Customer Churn in the IBM Telco Dataset  
### *STAT 159/259 Final Project Group 2*


## Group Members

**Marcel Gunadi**, **Ethan Chant**, **Benson Chang**, **Sophie Hanson**


## Project Description

This project analyzes the **IBM Telco Customer Churn dataset**, a simulated dataset describing a telecom company's customers. Our goal is to identify patterns associated with churn, build simple and interpretable predictive models, and derive insights that could inform targeted customer retention strategies. 

The IBM Telco Customer Churn dataset is publicly available from IBM. We do not claim ownership over this dataset.

Click the Binder badge to access an executable version of code from the project notebook: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-f25/final-group02.git/main?labpath=Project+Notebook.ipynb)

[View MyST website here](https://UCB-stat-159-f25.github.io/final-group02)

[GitHub repo](https://github.com/UCB-stat-159-f25/final-group02.git)


## Repository Structure

```
.
├── Data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── project_notebook.ipynb            # Full analysis including EDA and modeling
├── main.ipynb                         # Summary of the analysis
│
├── Output/                           # Rendered charts from project_notebook
│
├── utils.py                          # compute_accuracy, compute_auc
├── _toc.yml                          # Reproducible environment
├── myst.yml                          # Reproducible environment
├── environment.yml                   # Reproducible Conda environment
├── Makefile                          # Automation for running notebooks
├── README.md                         # Project documentation
└── LICENSE                           # Project license

```


## Installation

To clone the repository and enter the root directory, run these commands:
```bash
git clone https://github.com/UCB-stat-159-f25/final-group02.git
cd final-group02
```
To create the environment, run this command:
```bash 
conda env create -f environment.yml 
```
To run the environment, run this command:
```bash 
conda activate stat159-final
```


## License

This project is distributed under the terms specified in the `LICENSE` file.
