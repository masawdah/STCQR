<h3 align="center">Uncertainty Quantification for Deep Learning-Based Soil Moisture Forecasting Using Conformal Prediction</h3>

This repository contains the code used in the paper:  
**"Reliable Uncertainty Quantification for Deep Learning-Based Soil Moisture Forecasting Using Conformal Prediction."**

The code implements conformal prediction with spatial, temporal, and spatio-temporal localization to quantify uncertainty in soil moisture forecasting under exchangeability violations.

## Environment Setup (Conda)
Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate my-env

## Data Source
The soil moisture data were obtained from the International Soil Moisture Network (ISMN). ISMN integrates ground-based observations from multiple monitoring networks worldwide, ensuring high-quality and standardized soil moisture data.

## Reproducibility Note
Results may vary slightly when running the notebook in different environments or software setups (e.g., due to differences in library versions or hardware). However, the overall findings remain consistent: STCQR achieves coverage closer to the nominal level while maintaining sharper (narrower) prediction intervals.