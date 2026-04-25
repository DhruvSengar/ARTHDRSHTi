# Arth Drishti
## Overview
Arth Drishti is a descriptive economic analytics project analyzing India’s macroeconomic indicators (2000–2024) using World Bank data.
It focuses on extracting insights through data cleaning, statistical analysis, and strong visualizations, without any forecasting or machine learning.
---
## Features
* 7-step data cleaning pipeline (zero missing values)
* Descriptive statistics: mean, standard deviation, skewness, kurtosis, CAGR, volatility
* Economic feature engineering (inflation-adjusted growth, savings–investment balance, trade openness)
* Dimensional analysis using PCA and LDA (descriptive only)
* Multi-panel visualizations with annotated economic shocks (2008, 2020)
* Correlation analysis and comparative dashboards
---
## Dataset
* Source: World Bank Open Data (India)
* Time Range: 2000–2024
* Indicators include:
  * GDP, GDP Growth, CPI Inflation
  * Savings, Investment
  * Trade (Exports, Imports)
  * Population and Urbanisation
---
## Project Structure
```
Arth-Drishti/
│── main_analysis.py
│── arth_drishti_ui.py
│── india_wb_dirty_wide.csv
│── README.md
```
---
## How to Run
1. Clone the repository:
   git clone https://github.com/DhruvSengar/ARTHDRSHTi.git
2. Navigate into folder:
   cd ARTHDRSHTi
3. Install dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
4. Run the application:
   python arth_drishti_ui.py
---
## Visualizations
The project includes:
* GDP trends and growth analysis
* Inflation distribution and volatility plots
* Trade balance comparisons
* Savings vs Investment analysis
* Correlation heatmaps
* PCA and LDA plots
* Raw vs cleaned data comparison tables
---
## Key Insights
* India shows strong long-term GDP growth (2000–2024)
* Inflation volatility reduces after 2015
* Savings and investment remain closely aligned
* Trade deficit persists due to import dependence
* Urbanisation steadily increases over time
---
## Technologies Used
* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn (for PCA & LDA only)
---
## Note
This project is strictly descriptive and analytical. No forecasting, prediction, or machine learning models are used.
---
## Author
Dhruv Sengar
