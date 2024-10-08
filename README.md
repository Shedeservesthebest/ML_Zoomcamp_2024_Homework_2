# ML_Zoomcamp_2024_Homework_2
# Laptops Price Prediction - ML Zoomcamp 2024 Homework 2

This project is part of the **ML Zoomcamp 2024** course. The goal is to build a regression model to predict laptop prices based on a dataset that includes features such as RAM, storage, and screen size.

## Dataset

The dataset used for this project is the Laptops Price Dataset from [Kaggle](https://www.kaggle.com/datasets/juanmerinobermejo/laptops-price-dataset). It is available for download directly via the following link:

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv
Objective

The main objective of this homework is to create a regression model that predicts the laptop prices using selected features. We use the 'Final Price' column as the target variable.

Features Used

After cleaning and selecting the relevant features, we use the following columns:

ram: Laptop RAM in GB
storage: Storage in GB
screen: Screen size in inches
final_price: Final price of the laptop
Exploratory Data Analysis (EDA)

We perform EDA to understand the distribution of the target variable (final_price). Specifically, we examine whether it has a long tail or not.

Questions Addressed

Question 1: Missing Values
We determine which column has missing values.
Question 2: Median RAM
We calculate the median value for the ram column.
Question 3: Filling Missing Values
We deal with missing values by filling them with either 0 or the mean and train a regression model to evaluate which approach gives a better RMSE score.
Question 4: Regularized Linear Regression
We train a regularized linear regression model using various regularization parameters r and identify the value that provides the best RMSE.
Question 5: Impact of Seed Value on Model Stability
We investigate how different random seed values affect the model\'s RMSE by calculating the standard deviation of RMSE across multiple seeds.
Question 6: Final Test Evaluation
We train the final model with a regularization parameter r=0.001 and evaluate its performance on the test dataset.



 



