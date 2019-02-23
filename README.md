# Linear Regression to predict Online News Popularity

For more information, read Report.pdf (in Portuguese) or contact me.

### Atividades

* Preparation and loading of the dataset and implementation of LR;
* Feature scaling and Feature Selection;
* Removal of noise and changes in Target.

## LR implementation

gradient_descent, normal_equation cost_function, fit, predict, rmse e r2_score.

<p align="center">
  <img src="imgs/cf_inicial.png">
</p>

## Feature scaling
Rescaling, Mean normalisation, Standardization. 

<p align="center">
  <img src="imgs/cf_scaling1.png">
</p>

<p align="center">
  <img src="imgs/cf_best_alfa_scaling.png">
</p>

<p align="center">
  <img src="imgs/fc_scaling_compare.png">
</p>

## Noise/Outlier Removal

## Feature Selection

| UC | RFE | PCA 
--- | --- | --- |
| Num. features   |  58       |  31           |  54 | 
| Cost Function |  5*10^7   |  5.5*10^7     |  5.5*10^7 |  
| RMSE            |  8*10^3   |  7.8*10^3     |  7.8*10^3 |  

## Normal Equation
