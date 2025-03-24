# Bulldozer Price Prediction with Random Forest Regressor

## Overview
This repository contains a solution for predicting the sale prices of bulldozers using a Random Forest Regressor. The dataset is from a past Kaggle competition: **"Blue Book for Bulldozers"**, which challenges participants to develop models that estimate the auction prices of bulldozers based on historical data.

## Dataset
The dataset can be found on Kaggle: [Blue Book for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers/overview)

The dataset includes features such as:
- Equipment ID
- Sale date
- Machine hours usage
- Model description
- Location
- Year of manufacture
- Various categorical and numerical attributes describing the bulldozer

## Model & Approach
The model used in this project is **Random Forest Regressor**, a powerful ensemble learning technique that performs well on structured data. The steps involved include:

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature engineering
   - Splitting data into training and validation sets

2. **Model Training**
   - Training a Random Forest Regressor using Scikit-Learn
   - Hyperparameter tuning using GridSearchCV

3. **Model Evaluation**
   - Evaluating performance using RMSE (Root Mean Squared Error)
   - Visualizing feature importance

## Dependencies
To run the project, install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Required Libraries
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bulldozer-price-prediction.git
   cd bulldozer-price-prediction
   ```
2. Download the dataset from Kaggle and place it in the `data/` directory.
3. Run the notebook or script to train the model and make predictions:
   ```bash
   python train_model.py
   ```
4. View results and feature importance plots.

## Results
- The trained model provides competitive RMSE scores on validation data.
- Feature importance analysis highlights the most influential factors in predicting bulldozer prices.

## Acknowledgments
- Kaggle for the dataset and competition
- Scikit-Learn for machine learning utilities

## Future Improvements
- Implementing more advanced models (e.g., Gradient Boosting, XGBoost, LightGBM)
- Improving feature engineering techniques
- Using time-series forecasting methods

## License
This project is open-source under the MIT License.

