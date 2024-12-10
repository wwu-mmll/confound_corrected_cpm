![Logo](assets/img/CCCPM_medium.png)
# Confound-Corrected Connectome-Based Predictive Modeling (CCCPM)
[![GitHub Workflow Status](https://github.com/wwu-mmll/confound_corrected_cpm/actions/workflows/python-test.yml/badge.svg)](https://github.com/wwu-mmll/confound_corrected_cpm/actions/workflows/python-test.yml)
[![Coverage Status](https://coveralls.io/repos/github/wwu-mmll/cpm_python/badge.svg?branch=main)](https://coveralls.io/github/wwu-mmll/cpm_python?branch=main)
[![Github Contributors](https://img.shields.io/github/contributors-anon/wwu-mmll/cpm_python?color=blue)](https://github.com/wwu-mmll/cpm_python/graphs/contributors)
[![Github Commits](https://img.shields.io/github/commit-activity/y/wwu-mmll/cpm_python)](https://github.com/wwu-mmll/cpm_python/commits/main)

CCCPM is a newly developed Python toolbox designed specifically for researchers in psychiatry and neuroscience to 
perform connectome-based predictive modeling. This package offers a comprehensive framework for building predictive 
models from structural and functional connectome data, with a strong focus on methodological rigor, interpretability, 
confound control, and statistical robustness.

---
## Background

Network-based approaches are increasingly recognized as essential for understanding the complex relationships in brain connectivity that underlie behavior, cognition, and mental health. In psychiatry and neuroscience, analyzing structural and functional networks can reveal patterns associated with mental disorders, support individualized predictions, and improve our understanding of brain function. However, these analyses require robust tools that account for the unique challenges of connectome data, such as high dimensionality, variability, and the influence of confounding factors.

Despite the growing importance of connectome-based predictive modeling (CPM), there is currently no fully developed software package for performing these analyses. Existing options are limited to a few MATLAB scripts, which lack the flexibility, transparency, and rigor required to foster replicable research. CCCPM addresses this gap by providing a Python-based, flexible, and rigorously designed toolbox that encourages replicable analyses while allowing researchers to tailor their workflows to specific research questions.

---

## Overview

CCCPM was developed to address key challenges in connectome-based analyses, including optimizing model hyperparameters, controlling for confounding variables, and assessing the reliability of selected network features. This toolbox introduces novel methods, such as stability metrics for selected edges, and integrates well-established practices like nested cross-validation and permutation-based significance testing. By doing so, CCCPM provides a powerful and transparent tool for researchers aiming to explore brain networks' contributions to predictive models.

### Key Features

- **Hyperparameter Optimization**: Fine-tune model parameters, such as p-thresholds for edge selection, to achieve better predictive performance.
- **Confound Adjustment**: Use partial correlation methods during edge selection to rigorously control for covariates and confounding variables.
- **Residualization**: Remove the influence of confounds from connectome strengths to ensure cleaner data inputs.
- **Statistical Validation**: Assess model and edge-level significance using permutation-based testing, ensuring that findings are statistically robust.
- **Stability Metrics**: Evaluate the reliability of selected edges across iterations, improving the interpretability and reproducibility of identified networks.
- **Model Increment Analysis**: Quantify the unique contribution of connectome data to predictive models, helping to clarify their added value in prediction tasks.

---

## Why CCCPM?

Unlike existing CPM implementations, which are limited in scope and flexibility, CCCPM is designed to foster rigorous and replicable research. Its Python-based architecture ensures accessibility and compatibility with modern data science workflows, while its features address the specific challenges of connectome-based analyses. By offering a robust and transparent framework, CCCPM enables researchers to conduct analyses that are not only flexible and customizable but also reproducible and scientifically sound.

---

## Features in Detail

### **Data Imputation**
CCCPM includes methods to handle missing data effectively, ensuring that datasets with incomplete connectome information can still be utilized without introducing biases.

### **Nested Cross-Validation**
A nested cross-validation scheme is implemented to separate hyperparameter tuning from model evaluation. This ensures that the reported model performance is unbiased and reflects its true generalization capability.

### **Threshold Optimization**
The toolbox automates the optimization of p-thresholds, which determine which edges in the connectome are selected for model building. This allows researchers to identify thresholds that balance performance and interpretability.

### **Confound Adjustment**
By implementing partial correlations, CCCPM allows researchers to account for confounding variables during edge selection, ensuring that identified networks represent genuine relationships rather than artifacts.

### **Statistical Significance**
Permutation-based testing is provided to evaluate the significance of both model performance and selected edges, adding rigor to findings and reducing the risk of false-positive results.

### **Edge Stability**
CCCPM introduces a stability metric for selected edges, helping researchers evaluate the consistency of their findings across multiple iterations. This enhances the reliability of results and their potential for replication.

### **Model Increment Analysis**
Assess the added predictive value of connectome data by calculating the incremental contribution of network features to overall model performance.

---