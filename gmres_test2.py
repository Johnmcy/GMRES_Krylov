import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres, LinearOperator
import matplotlib.pyplot as plt

# Define a Hilbert matrix
n = 100  # Set matrix size
A_dense = np.array([[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)])
A_sparse = csc_matrix(A_dense)

# Define a right-hand side such that the solution is all ones
b = A_sparse.dot(np.ones(n))

# Define a preconditioner (here we use a simple diagonal preconditioner)
P_inv = np.diag(1 / A_sparse.diagonal())
P_inv = csc_matrix(P_inv)
M = LinearOperator((n, n), matvec=P_inv.dot)

# List to store residuals at each iteration
residuals = []

# Iteration counter
iteration_counter = [0]

# Define the callback function
def callback(residual):
    iteration_counter[0] += 1  # Increment the counter
    residuals.append(residual)
    print(f"Iteration {iteration_counter[0]}: Residual = {residual}")

# Set a small convergence tolerance
tolerance = 1e-10

# Set restart parameter
restart = 20

# Solve the system with GMRES, with preconditioner and restart
x, exitCode = gmres(A_sparse, b, M=M, restart=restart, tol=tolerance, callback=callback)

# Output the exit code
print(exitCode)  # 0 indicates successful convergence

# Output the total number of iterations
total_iterations = iteration_counter[0]
print(f"Total number of iterations: {total_iterations}")

# Calculate and output the total number of restarts
total_restarts = total_iterations // restart
print(f"Total number of restarts: {total_restarts}")

# Check if the solution is close to the true solution
print(np.allclose(A_sparse.dot(x), b))

# Plot the residual decay
plt.figure()
plt.semilogy(residuals, marker='o', linestyle='-')
plt.title('Residual Decay')
plt.xlabel('Iteration Number')
plt.ylabel('Residual')
plt.grid(True)
plt.show()
