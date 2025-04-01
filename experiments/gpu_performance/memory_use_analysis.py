
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the CSV file (replace 'data.csv' with your filename)
df = pd.read_csv("memory_stats.csv")

# Assume the first 4 columns are independent variables and the fifth column is the dependent variable.
# Column headings are
# num views, num rows, num channels, batch size, peak memory (GB), avail memory (GB), elapsed time0 (sec), elapsed time1 (sec)
PARAM_VALUES = [50, 100, 250, 500, 800, 1250]
BATCH_SIZES = [100, 250, 500, 800]

independent_vars = ['num views', 'num rows', 'num channels', 'batch size']

# --- Exploratory Data Analysis (EDA) ---

# Choose two independent variables (for example, 'var1' and 'var2')
# and the dependent variable (e.g., 'dependent')
dependent_var = 'elapsed time0 (sec)'  # 'elapsed time0 (sec)'  # 'peak memory (GB)'
var1 = 'num rows'
var2 = 'num views'
var3 = 'num channels'
df = df[(df['avail memory (GB)'] > 40) & (df['avail memory (GB)'] < 41)]
# df = df[(df['batch size'] == 500)]
df = df[df[var1] >= 50]
df = df[df[var2] == 500]
df = df[df[var3] == 250]
x = df[var1]
y = df[var2]
c = df[var3]
z = df[dependent_var]

# # Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the points
# sc = ax.scatter(x, y, z, c=c, cmap='viridis', marker='o')
#
# # Set labels
# ax.set_xlabel(var1)
# ax.set_ylabel(var2)
# ax.set_zlabel(dependent_var)
#
# # Add a colorbar to indicate the dependent variable's values
# plt.colorbar(sc, label=var3)
# plt.title("3D Scatter Plot: " + dependent_var + " vs " + var1 + " and " + var2)
#
# # Display the plot
# plt.show()
# # 1. Plot a histogram of the dependent variable to check its distribution (look for floor effects)
# plt.figure(figsize=(8, 4))
# plt.hist(y, bins=30, edgecolor='black')
# plt.title("Histogram of Dependent Variable")
# plt.xlabel("Dependent Value")
# plt.ylabel("Frequency")
# plt.show()
#
# # 2. Create scatter plots for each independent variable against the dependent variable
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# axs = axs.ravel()
# for i, col in enumerate(independent_vars):
#     axs[i].scatter(X[col], y, alpha=0.7)
#     axs[i].set_title(f"{col} vs Dependent")
#     axs[i].set_xlabel(col)
#     axs[i].set_ylabel("Dependent")
# plt.tight_layout()
# plt.show()

# 3. Optionally, view the correlation matrix
corr_matrix = df[[*independent_vars, dependent_var]].corr()
print("Correlation Matrix:")
print(corr_matrix)

# --- Regression Modeling using scikit-learn ---

# Initialize and fit the linear regression model

X = df[independent_vars]
y = df[dependent_var]

model = LinearRegression()
model.fit(X, y)

# Get predictions
predictions = model.predict(X)

# Calculate R-squared
r_squared = r2_score(y, predictions)

# Display the model parameters
print("Intercept:", model.intercept_)
print("Coefficients:")
for var, coef in zip(independent_vars, model.coef_):
    print(f"  {var}: {coef}")
print("R-squared:", r_squared)

# --- Residual Analysis ---

# Calculate residuals
residuals = y - predictions

# Plot residuals vs fitted values to check model assumptions
plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals, c=df['batch size'])
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predictions")
plt.ylabel("Residuals")
plt.title("Residuals vs Predictions")
plt.show()

# If floor effects are suspected, consider additional modeling adjustments,
# such as transforming the dependent variable or using a non-linear model.