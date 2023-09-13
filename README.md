# Restaurant Sales Prediction

This repository contains code for predicting restaurant sales using various machine learning models. The code is organized into several sections, including data analysis, data preprocessing, model training, and evaluation.

## Code Structure

1. **Data Analysis:** 
   - Data analysis is performed on three datasets: `meal_info.csv`, `fulfilment_center_info.csv`, and `train.csv`.
   - The analysis includes exploring data statistics, checking for missing values, and visualizing trends using libraries like Pandas, NumPy, Matplotlib, and Seaborn.

2. **Data Preprocessing:** 
   - Data preprocessing involves tasks such as feature engineering, merging datasets, encoding categorical variables, and preparing the data for model training.
   - Outliers can be removed (optional) during preprocessing.
   - The final preprocessed data is saved in `train_feature.csv` and `test_feature.csv` files.

3. **Model Training:** 
   - Several machine learning models are trained, including Decision Tree Regressor, K-Nearest Neighbors Regressor, RandomForest Regressor, AdaBoost Regressor, LightGBM Regressor, and XGBoost Regressor.
   - Hyperparameter tuning is performed using GridSearchCV for the LightGBM Regressor.
   - Models are evaluated using the Root Mean Squared Logarithmic Error (RMSLE) and R-squared (R^2) score.

4. **Feature Importance:** 
   - Feature importance is visualized for the LightGBM Regressor.

5. **Submission:** 
   - The final ensemble model, which combines predictions from LightGBM and XGBoost, is used to generate sales predictions for the test dataset.
   - The predictions are saved in `ensemble.csv` for submission.

## Data Files

- `meal_info.csv`: Information about meals, including category and cuisine.
- `fulfilment_center_info.csv`: Information about fulfilment centers, including location and type.
- `train.csv`: Training dataset with information about orders and sales.
- `train_feature.csv`: Preprocessed training dataset with features.
- `test_feature.csv`: Preprocessed test dataset with features.
- `ensemble.csv`: Final sales predictions for the test dataset.

## Dependencies

Make sure you have the following Python libraries installed:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- LightGBM
- XGBoost
- Scikit-learn

## How to Use

1. Clone this repository to your local machine.

2. Ensure you have the required dependencies installed.

3. Run the code sections in the provided order (`Data Analysis`, `Data Preprocessing`, `Model Training`, `Feature Importance`, and `Submission`).

4. The final sales predictions will be saved in the `ensemble.csv` file.

## Notes

- This code is intended for educational purposes and may require further customization for specific use cases.
- Hyperparameters for the machine learning models can be fine-tuned for better performance.

Feel free to reach out if you have any questions or need further assistance.
