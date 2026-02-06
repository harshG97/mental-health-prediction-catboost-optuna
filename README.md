# 🧠 Depression Prediction with CatBoost & Optuna

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Latest-orange.svg)](https://catboost.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-green.svg)](https://optuna.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning pipeline for depression prediction using **CatBoost** with **Optuna** hyperparameter optimization and extensive feature engineering. This project demonstrates best practices in handling categorical data, missing values, and model optimization for Kaggle competitions.

See and run this notebook on [Kaggle](https://www.kaggle.com/code/harshg97/catboost-optuna-tuning-with-feature-engineering/output?scriptVersionId=296143083).

## 📊 Project Overview

This notebook tackles a binary classification problem (depression prediction) using a sophisticated ML pipeline that includes:

- **Smart Data Integration**: Merging competition and original datasets for enhanced training
- **Advanced Feature Engineering**: Creating interaction features, ratio features, and categorical binning
- **Automated Hyperparameter Tuning**: Using Optuna's TPE algorithm for optimal CatBoost parameters
- **Robust Cross-Validation**: Stratified K-Fold CV with ensemble predictions
- **Production-Ready Code**: Clean, documented, and reproducible

### 🎯 Key Results

- Effective handling of **high-cardinality categorical features**
- **Engineered features** (PS_ratio, Age_bin, PF_factor) rank among top feature importances
- **Optuna-tuned** hyperparameters provide optimal model performance
- **5-fold CV ensemble** reduces overfitting and improves generalization

## 🗂️ Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Model Details](#model-details)
- [Results](#results)
- [Key Learnings](#key-learnings)
- [Contributing](#contributing)
- [License](#license)

## Features

### Data Processing
- ✅ Intelligent merging of multiple data sources
- ✅ Systematic handling of high-cardinality categorical features
- ✅ Three-strategy approach to missing value imputation
- ✅ Automated column name standardization and string cleaning

### Feature Engineering
- 🔧 **Consolidation**: Merging mutually exclusive features (Pressure, Satisfaction)
- 🔧 **Ratio Features**: Creating interaction terms (PS_ratio, PSF_ratio, PF_factor)
- 🔧 **Smart Binning**: Age categorization with equal-sized bins
- 🔧 **Frequency-Based Filtering**: Grouping rare categories to reduce noise

### Model Training
- 🚀 **CatBoost**: Native categorical feature handling
- 🚀 **Optuna**: Automated hyperparameter optimization (100 trials)
- 🚀 **Stratified K-Fold**: 5-fold CV maintaining class distribution
- 🚀 **Ensemble**: Averaging predictions across folds

## Installation

### Prerequisites

```bash
Python 3.8+
```

### Required Libraries

```bash
pip install numpy pandas matplotlib seaborn
pip install catboost optuna
pip install scikit-learn
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/depression-prediction-catboost.git
cd depression-prediction-catboost

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook catboost-optuna-tuning-enhanced.ipynb
```

## Usage

### Basic Workflow

1. **Load Data**
```python
competition = pd.read_csv('train.csv')
original = pd.read_csv('train_original.csv')
test = pd.read_csv('test.csv')
```

2. **Feature Engineering**
```python
# Process high-cardinality features
keep_desired_features(df_combined, df_test, desired_features, 'Profession')

# Create interaction features
df['PS_ratio'] = df['Pressure'] / df['Satisfaction']
df['Age_bin'] = pd.qcut(df['Age'], 15, duplicates='drop')
```

3. **Train Model**
```python
# Optuna hyperparameter tuning (if needed)
study = optuna.create_study(direction='maximize')
study.optimize(catboost_objective, n_trials=100)

# Train with best parameters
model = CatBoostClassifier(**best_params)
model.fit(X_train, y_train, cat_features=cat_cols)
```

4. **Generate Predictions**
```python
# Average predictions across CV folds
final_predictions = preds_test.mean(axis=1)
```

### Configuration

Toggle hyperparameter retuning:
```python
RETUNE_CATBOOST = False  # Set to True to re-run Optuna optimization
```

## Project Structure

```
depression-prediction-catboost/
│
├── catboost-optuna-tuning-enhanced.ipynb  # Main notebook with full pipeline
├── ENHANCEMENT_SUMMARY.md                 # Documentation of enhancements
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── data/                                  # Data directory (not included)
│   ├── train.csv
│   ├── train_original.csv
│   └── test.csv
│
└── outputs/                               # Model outputs and submissions
    ├── submission.csv
    └── feature_importance.png
```

## Methodology

### 1. Data Integration

**Challenge**: Competition data might not capture full distribution of real-world scenarios.

**Solution**: Merge competition and original datasets, tracking source for validation.

```python
df_combined = pd.concat([original, competition], ignore_index=True)
df_combined['source'] = ['original'] * len(original) + ['competition'] * len(competition)
```

### 2. Exploratory Data Analysis

- Bivariate analysis for numerical features (correlation plots)
- Categorical feature analysis (depression rates by category)
- Missing value pattern identification
- Feature correlation assessment

### 3. Preprocessing Pipeline

#### High-Cardinality Feature Handling
```python
def keep_desired_features(series_combined, series_test, desired_features, replacement='Other'):
    """
    Keep only frequent categories (50+ occurrences), group rare values.
    
    Why 50? Balance between:
    - Retaining signal from meaningful categories
    - Reducing noise from rare values
    - Preventing overfitting to specific instances
    """
```

#### Missing Value Strategy Matrix

| Feature Type | Strategy | Reason |
|-------------|----------|---------|
| Categorical (low missingness) | Mode | Random missingness likely |
| Numerical | Median | Robust to outliers |
| Categorical (high missingness) | Separate "None" category | Missingness is informative |
| CGPA | Conditional (median for students, "None" for workers) | Structural missingness |

### 4. Feature Engineering Deep Dive

#### Consolidation Features
```python
# Combine mutually exclusive features
df['Pressure'] = df.apply(lambda row: 
    row['AcademicPressure'] if row['WorkingProfessional'] == 'No' 
    else row['WorkPressure'], axis=1)

df['Satisfaction'] = df.apply(lambda row:
    row['StudySatisfaction'] if row['WorkingProfessional'] == 'No'
    else row['JobSatisfaction'], axis=1)
```

**Rationale**: Students have academic pressure/study satisfaction, workers have work pressure/job satisfaction. Never both. Consolidating eliminates sparsity.

#### Interaction Features
```python
# Capture relationships between stressors
df['PS_ratio'] = df['Pressure'] / df['Satisfaction']
df['PSF_ratio'] = (df['Pressure'] + df['Satisfaction']) / df['FinancialStress']
df['PF_factor'] = df['Pressure'] / df['FinancialStress']
```

**Rationale**: Depression isn't just about absolute stress levels—it's about the *balance* between pressures and fulfillment. Ratios capture this interaction.

#### Age Binning
```python
# Create 15 equal-frequency bins
df['Age_bin'] = pd.qcut(df['Age'], 15, duplicates='drop')
```

**Rationale**: Depression-age relationship is non-linear. Different life stages (teens, young adults, middle-age, seniors) have different risk profiles. Binning lets the model learn stage-specific patterns.

## Model Details

### Why CatBoost?

1. **Native Categorical Handling**: No need for one-hot or label encoding
2. **Ordered Boosting**: Reduces prediction shift and overfitting
3. **Robust to Overfitting**: Built-in regularization mechanisms
4. **Missing Value Support**: Handles NaNs naturally
5. **Fast Training**: GPU support and efficient algorithms

### Hyperparameter Optimization with Optuna

**Search Space**:
```python
{
    'iterations': [500, 2000],
    'learning_rate': [0.01, 0.3],
    'depth': [4, 10],
    'l2_leaf_reg': [1, 10],
    'border_count': [32, 255],
    'random_strength': [1, 10]
}
```

**Optimization Strategy**:
- Algorithm: Tree-structured Parzen Estimator (TPE)
- Trials: 100
- Objective: Maximize CV accuracy
- Pruning: MedianPruner (stop unpromising trials early)

### Cross-Validation Strategy

**Stratified 5-Fold CV**:
- Maintains class distribution in each fold
- Trains 5 models on different 80/20 splits
- Final prediction = average of 5 models
- Reduces variance and overfitting

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    # Train on 80%, validate on 20%
    # Store predictions for ensemble
```

## Results

### Feature Importance Top 10

Based on CatBoost's feature importance analysis:

1. **Age_bin** ⭐ (Engineered)
2. **SleepDuration**
3. **DietaryHabits**
4. **PS_ratio** ⭐ (Engineered)
5. **FinancialStress**
6. **Pressure** ⭐ (Engineered)
7. **Have you ever had suicidal thoughts?**
8. **PF_factor** ⭐ (Engineered)
9. **Satisfaction** ⭐ (Engineered)
10. **Name_processed**

⭐ = Engineered features

**Key Insight**: 5 of the top 10 features are engineered, validating our feature engineering strategy!

### Model Performance

- **Cross-Validation**: Consistent performance across folds
- **Ensemble**: Averaging reduces prediction variance
- **Threshold Optimization**: Explored 0.5, 0.8, and probability-based thresholds

### The Name Column Mystery 🔍

**Observation**: The `Name` column shows unexpected predictive power.

**Explanation**: Likely a synthetic data artifact in Kaggle Playground competitions. Name generation might have inadvertently embedded patterns correlated with the target.

**Recommendation**: 
- ✅ Use for competition (improves score)
- ❌ Remove in production (won't generalize)
- 🔍 Always validate feature importance makes domain sense

## Key Learnings

### 1. Feature Engineering Beats Complex Models
Thoughtful feature engineering (ratios, consolidation, binning) provided more value than hyperparameter tuning alone.

### 2. Handle Categorical Features Smartly
- Grouping rare categories reduces noise
- Frequency-based filtering prevents overfitting
- Native categorical handling (CatBoost) > one-hot encoding

### 3. Missing Values Require Strategy
Not all missing values are equal:
- Random missingness → Impute
- Structural missingness → Conditional imputation
- Informative missingness → Separate category

### 4. Cross-Validation is Your Friend
- Stratified K-Fold maintains class balance
- Ensemble averaging improves robustness
- Multiple models > single model

### 5. Watch for Data Leakage
Unexpected feature importance (like Name) can reveal:
- Synthetic data artifacts
- Data leakage
- Invalid assumptions

### 6. Automation Saves Time
- Optuna automates hyperparameter search
- Reusable functions for feature processing
- Systematic approach > manual trial-and-error

## Future Improvements

- [ ] **Ensemble with other models**: XGBoost, LightGBM, Neural Networks
- [ ] **Advanced feature engineering**: Polynomial features, clustering-based features
- [ ] **Automated feature selection**: Recursive feature elimination, permutation importance
- [ ] **Deep learning**: Explore TabNet or other deep learning architectures
- [ ] **Explainability**: SHAP values for model interpretation
- [ ] **Production pipeline**: MLflow for experiment tracking, model versioning

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [CatBoost Documentation](https://catboost.ai/docs/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle Playground Series](https://www.kaggle.com/competitions)

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Built with ❤️ for the machine learning community*
