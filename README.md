<h1 align="center">Welcome to data-solutions-capstone - Water Potability Analysis 👋</h1>
<p>
  <img src="https://img.shields.io/badge/python-%3E%3D3.9.0-blue" />
</p>

> Assessing the effects that twenty different biological and chemical characteristics of water have on water potability.

### 🏠 [Homepage](https://github.com/LucyDykhouse/data-solutions-capstone)  

### ✨ [Demo](Link here)

### Hosted Jupyter Notebooks
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LucyDykhouse/data-solutions-capstone/main)

## Prerequisites

- python >= 3.9.0

## Install

```sh
pip install -r requirements.txt
```

## Notebook Usage
```sh
jupyter notebook
```
Or consult hosted notebooks here [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LucyDykhouse/data-solutions-capstone/main)

## Dashboard Usage
```sh
python3 hypothesis_dashboard.py
```

## File Structure
**data/** : contains initial dataset along with clean dataset and engineered dataset ready for analysis  
**models/** : contains exported logistic regression models (one with all features, one with relevant features)  

**Project Overview.pdf** : lists background information, project motivation, and hypothesis  
**Dataset Description.pdf** : includes data sources and data dictionary  

**Data Exploration.ipynb** : displays summary statistics, basic visualizations, and correlations  
**Data Wrangling.ipynb** : prepares data for statistical analysis through cleansing and feature engineering  
**Statistical Analysis and Hypothesis Testing.ipynb** : Implements logistic regression and analyzes outputs  

**hypothesis_dashboard.py** : generates interactive dashboard to display hypothesis testing results  
**requirements.txt** : lists packages and versions installed  

## Acceptance Criteria
As a user, I should be able to infer which factors most affect water quality using data visualizations.  
As a user, I should be able to see various property values that act as thresholds for whether water is potable using data visualizations.  
As a user, I should be able to tell which factors most affect water quality using quantitative values from univariate/multivariate regressions.  

## Author

👤 **Lucy Dykhouse**

* GitHub: [@LucyDykhouse](https://github.com/LucyDykhouse)

## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_
