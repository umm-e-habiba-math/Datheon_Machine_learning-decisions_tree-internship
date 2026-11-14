# Datheon_Machine_learning-decisions_tree-internship
### Task 1: Multi-Class Benchmark Classification Pipeline (Iris Dataset)
  Objective: Built an end-to-end supervised classification pipeline using Logistic Regression and Decision Trees on the Iris dataset.
  Methodology: Handled feature standardization (StandardScaler), 80/20 train-test splitting, cross-evaluation, and plotted confusion matrix heatmaps.
  Results: Achieved 100% precision, recall, and F1-score performance across all target species ('setos', 'versicolor','virginica').
  # Task 2: Real Estate Housing Price Prediction (California Housing Dataset)
 Objective: Built regularized linear regression models (Ridge and Lasso) to predict California median housing values based on structural, geographical, and demographic features.
 Methodology: Imputed missing values (207 missing entries in 'total_bedrooms') using column median, converted ocean_proximity to numeric indicators via One-Hot Encoding ('pd.get_dummies'), scaled features using 'StandardScaler', and split data 80/20.
 Results: Achieved an $R^2$ score of ~0.6255 and RMSE of ~$70,057 on test set predictions.
