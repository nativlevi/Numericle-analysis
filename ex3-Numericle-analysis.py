# neria atias 316118728
# nativ levi 209150879
# https://github.com/neriatias/Numerical-analysis



import numpy as np


def jacobi_method(coefficients, constants, initial_guess, tolerance=0.001, max_iterations=50):
    n = len(coefficients)
    x = initial_guess.copy()
    x_new = np.zeros(n)
    iteration = 0

    while iteration < max_iterations:
        for i in range(n):
            sum_ax = np.dot(coefficients[i, :i], x[:i]) + np.dot(coefficients[i, i + 1:], x[i + 1:])
            x_new[i] = (constants[i] - sum_ax) / coefficients[i, i]

        if np.allclose(x, x_new, atol=tolerance):
            print(f'Jacobi method converged in {iteration + 1} iterations.')
            return x_new

        x = x_new.copy()
        iteration += 1

    print('Jacobi method did not converge within the specified tolerance and maximum iterations.')
    return x_new


def gauss_seidel_method(coefficients, constants, initial_guess, tolerance=0.001, max_iterations=50):
    n = len(coefficients)
    x = initial_guess.copy()
    iteration = 0

    while iteration < max_iterations:
        for i in range(n):
            sum_ax = np.dot(coefficients[i, :i], x[:i]) + np.dot(coefficients[i, i + 1:], x[i + 1:])
            x[i] = (constants[i] - sum_ax) / coefficients[i, i]

        if np.linalg.norm(np.dot(coefficients, x) - constants) < tolerance:
            print(f'Gauss-Seidel method converged in {iteration + 1} iterations.')
            return x

        iteration += 1

    print('Gauss-Seidel method did not converge within the specified tolerance and maximum iterations.')
    return x


def main():
    # Example 3x3 matrix
    coefficients = np.array([[10, -1, 2],
                             [-1, 11, -1],
                             [2, -1, 10]], dtype=float)
    constants = np.array([6, 25, -11], dtype=float)
    initial_guess = np.array([0, 0, 0], dtype=float)

    # Check for diagonal dominance
    if is_diagonally_dominant(coefficients):
        # Solve using Jacobi method
        print("Using Jacobi method:")
        jacobi_solution = jacobi_method(coefficients, constants, initial_guess)
        print("Solution:", jacobi_solution)

        # Solve using Gauss-Seidel method
        print("\nUsing Gauss-Seidel method:")
        gauss_seidel_solution = gauss_seidel_method(coefficients, constants, initial_guess)
        print("Solution:", gauss_seidel_solution)
    else:
        print("Matrix is not diagonally dominant. Cannot guarantee convergence of Jacobi or Gauss-Seidel methods.")


def is_diagonally_dominant(matrix):
    n = len(matrix)
    for i in range(n):
        row_sum = np.sum(np.abs(matrix[i])) - np.abs(matrix[i, i])
        if np.abs(matrix[i, i]) <= row_sum:
            return False
    return True


if __name__ == "__main__":
    main()
