import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt

# Define the matrix and right-hand side
A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
b = np.array([2, 4, -1], dtype=float)

# List to store residuals at each iteration
residuals = []

# Iteration counter
iteration_counter = [0]

# Define the callback function
def callback(residual):
    iteration_counter[0] += 1  # Increment the counter
    residuals.append(residual)
    print(f"Iteration {iteration_counter[0]}: Residual = {residual}")

# Set the convergence tolerance
tolerance = 1e-10

# Solve the system with GMRES
x, exitCode = gmres(A, b, tol=tolerance, callback=callback)

# Output the exit code
print(exitCode)  # 0 indicates successful convergence

# Output the total number of iterations
print(f"Total number of iterations: {iteration_counter[0]}")

# Check if the solution is close to the true solution
print(np.allclose(A.dot(x), b))

# Plot the residual decay
plt.figure()
plt.semilogy(residuals, marker='o', linestyle='-')
plt.title('Residual Decay')
plt.xlabel('Iteration Number')
plt.ylabel('Residual')
plt.grid(True)
plt.show()
