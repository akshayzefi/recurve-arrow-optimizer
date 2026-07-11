# Recurve Arrow Design Optimizer

## Overview
A machine learning-based optimization framework that predicts recurve arrow performance (speed and grouping accuracy) and recommends optimal design configurations, replacing traditional trial-and-error tuning with a data-driven approach. Built as an interactive Streamlit application, it lets archers, coaches, and equipment designers tune arrow parameters and instantly see predicted performance.

**Live Demo:** [recursive-arrow-optimizer.streamlit.app](https://recursive-arrow-optimizer-hme5mb4d5n8yyurx8s7yip.streamlit.app/)

## Problem Statement
Recurve arrow performance depends heavily on design parameters like length, point mass, shaft density, and fletching area — small variations affect both speed and accuracy. Traditional tuning relies on manual, time-intensive trial-and-error. This project uses machine learning to predict performance instantly and Bayesian optimization to search for the best parameter combinations automatically.

## Approach

### Model
- **Algorithm:** Multi-Output Regressor wrapped around XGBoost, predicting velocity and group size simultaneously
- **Why XGBoost:** captures nonlinear relationships and threshold effects in arrow flight dynamics (e.g., fletching area improves stability only up to a point, then adds drag) that linear models miss
- **Preprocessing:** StandardScaler fitted on training data only (prevents data leakage), 80/20 train-test split with fixed random seed for reproducibility

### Hyperparameters
| Parameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |

### Model Performance

| Metric | Velocity | Group Size |
|---|---|---|
| R² | 0.9978 | 0.9693 |
| MSE | 0.0796 | 4.8643 |
| MAE | 0.2182 m/s | 1.7538 mm |

The model explains 99.78% of variance in velocity and 96.93% in group size, with predictions accurate to within ~0.22 m/s and ~1.75 mm on average.

### Optimization
Bayesian optimization searches the design space efficiently (fewer evaluations than grid/random search) across three user-selectable modes:
- **Balanced Profile** — equal weight to speed and accuracy
- **Velocity Mode** — prioritizes maximum speed
- **Precision Mode** — prioritizes minimum group size

## Example Result

Optimization improved a baseline design significantly:

| | Before Optimization | After Optimization |
|---|---|---|
| Length | 650 mm | 595 mm |
| Point Mass | 7.5 g | 9.6 g |
| Shaft Density | 0.00015 g/mm³ | 0.000104 g/mm³ |
| Fletching Area | 900 mm² | 920 mm² |
| Velocity | 72.8 m/s | 76.0 m/s |
| Group Size | 23.2 mm | 3.4 mm |
| FOC Balance | 40.0% | 55.4% |

Group size dispersion improved by nearly 7x while velocity also increased — demonstrating the value of multi-objective optimization over single-parameter manual tuning.

## Dataset
- 5,000 simulated arrow configurations generated from physics-based equations and archery dynamics literature
- 4 input parameters: length (mm), point mass (g), shaft density (g/mm³), fletching area (mm²)
- 2 primary targets: velocity (m/s), group size (mm); 2 secondary metrics: total mass (g), FOC balance (%)
- Split: 4,000 training samples (80%), 1,000 test samples (20%)

## Repository Structure
- MOO_Recurve_Archery.ipynb
- app2.py
- README.md
- requirements.txt
- .gitignore


## Setup

```bash
git clone https://github.com/akshayzefi/recurve-arrow-optimizer.git
cd recurve-arrow-optimizer
pip install -r requirements.txt
streamlit run app2.py
```

## Requirements
streamlit
xgboost
scikit-learn
pandas
numpy
scikit-optimize
matplotlib


## Key Learnings
- Multi-output regression captures interdependencies between correlated targets (velocity and group size) better than training separate models
- Bayesian optimization finds near-optimal design configurations with far fewer evaluations than brute-force grid search
- Feature scaling must be fitted only on training data to avoid leaking test-set information into preprocessing

## Limitations
- Dataset is synthetic/simulated — real-world physical validation is still required
- Environmental factors like wind and humidity are not modeled
- Framework currently limited to recurve archery; not validated for compound or barebow setups

## Future Scope
- Incorporate real-world testing data for validation
- Add environmental parameters (wind, humidity, temperature)
- Explore deep learning models for more complex pattern recognition
- Extend framework to other archery disciplines

## Author
Akshay Das — BE (SW) Production Engineering, PSG College of Technology
