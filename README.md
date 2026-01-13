# Port Equipment Predictive Maintenance Model

A machine learning solution for predicting maintenance needs of port equipment using Random Forest classification.

## Overview

This project implements a predictive maintenance system for port infrastructure, analyzing various operational parameters to determine when equipment maintenance is required. The model helps port operators optimize maintenance schedules, reduce downtime, and prevent equipment failures.

## Features

- **Predictive Maintenance Classification**: Binary classification to predict whether equipment needs maintenance
- **Feature Importance Analysis**: Identify which factors most influence maintenance needs
- **Data Visualization**: Comprehensive visualizations including:
  - Feature importance bar charts
  - Equipment age distribution histograms
  - Power consumption boxplots
  - Load capacity vs. vibration scatter plots
  - Time series analysis of maintenance patterns

## Dataset Features

The model uses the following operational parameters:

| Feature | Description |
|---------|-------------|
| `power_consumption` | Equipment power usage (kWh) |
| `temperature` | Ambient temperature (°C) |
| `humidity` | Environmental humidity (%) |
| `equipment_age_days` | Age of equipment in days |
| `operational_hours` | Daily operational hours |
| `load_capacity_percentage` | Current load capacity (%) |
| `voltage_variation` | Voltage fluctuation levels |
| `power_factor` | Electrical power factor |
| `vibration_level` | Equipment vibration measurements |
| `number_of_ships_berthed` | Ships currently at port |
| `hour`, `day_of_week`, `month` | Temporal features |

## Maintenance Trigger Conditions

Equipment is flagged for maintenance when:

- High power consumption (>2000 kWh) combined with old equipment (>5 years)
- Low power factor (<0.85)
- High vibration (>2.5) with high load capacity (>90%)
- Voltage variation exceeds ±8%
- Equipment age exceeds 8+ years
- Extended operation (>20 hours) with high load (>85%)

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Open the Jupyter notebook `port_maintenance_MLMODEL-.ipynb`
2. Run all cells sequentially
3. Review the classification report for model performance
4. Analyze visualizations to understand maintenance patterns

## Model Details

- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 trees
- **Preprocessing**: StandardScaler for feature normalization
- **Train/Test Split**: 80/20

## Output

The model provides:
- Classification report with precision, recall, and F1-score
- Feature importance rankings
- Multiple data visualizations for exploratory analysis

## Project Structure

```
├── port_maintenance_MLMODEL-.ipynb   # Main notebook
├── README.md                          # This file
```

## Future Enhancements

- Integration with real-time sensor data
- Model deployment as REST API
- Additional ML algorithms comparison
- Hyperparameter tuning
- Cross-validation implementation

## License

This project is for educational and research purposes.
