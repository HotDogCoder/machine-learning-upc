import numpy as np
import matplotlib.pyplot as plt

# Creating a synthetic dataset for linear regression
# Setting a random seed for reproducibility
np.random.seed(42)

# Generate 100 random x values between 0 and 10
x = np.linspace(0, 10, 100)
# Generate corresponding y values with a slope of 2.5 and some noise
y = 2.5 * x + np.random.normal(0, 1, x.size)

# Plotting the generated data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.title("Synthetic Linear Regression Dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()