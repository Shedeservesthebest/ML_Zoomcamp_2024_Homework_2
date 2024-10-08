# ML_Zoomcamp_2024_Homework_2
markdown
Copy code
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
We investigate how different random seed values affect the model's RMSE by calculating the standard deviation of RMSE across multiple seeds.
Question 6: Final Test Evaluation
We train the final model with a regularization parameter r=0.001 and evaluate its performance on the test dataset.
Setup and Dependencies

1. Clone the repository
bash
Copy code
git clone https://github.com/Shedeservesthebest/ML_Zoomcamp_2024_Homework_2.git
2. Install Required Libraries
Ensure that you have the necessary Python libraries installed. You can use the following requirements.txt to install them.

bash
Copy code
pip install -r requirements.txt
Here’s an example of a requirements.txt file:

Copy code
pandas
numpy
scikit-learn
3. Run the Notebook
After installing the dependencies, run the Jupyter notebook or Python script to perform the analysis.

Steps to Reproduce
Download the dataset using the link above.
Normalize the column names and filter the relevant columns (ram, storage, screen, final_price).
Perform the Exploratory Data Analysis (EDA) to check for missing values and variable distributions.
Split the data into training, validation, and test sets with a 60%/20%/20% ratio.
Train a linear regression model on the dataset and evaluate using RMSE.
For Question 4, try different regularization values (r) and find the best.
For Question 5, explore the effect of changing the random seed on the model's stability by calculating the standard deviation of the RMSE values.
Train the final model using all data (train and validation) and evaluate it on the test set.
Results

RMSE Scores
RMSE with missing values filled with 0: <your RMSE>
RMSE with missing values filled with mean: <your RMSE>
Best RMSE for regularized linear regression: <r value>
Final Test RMSE: <your RMSE>
Author

This homework is part of ML Zoomcamp 2024 by [Your Name].

License

This project is licensed under the MIT License - see the LICENSE file for details.

markdown
Copy code

### Steps to Add the README to GitHub:

1. **Create the README File**:
   - In your local project directory, create a file named `README.md`.

   ```bash
   touch README.md
Edit the README:
Open the README.md file in VS Code and paste the content (or adjust it based on your specific project).
Stage, Commit, and Push the README:
bash
Copy code
git add README.md
git commit -m "Added README file"
git push origin main
