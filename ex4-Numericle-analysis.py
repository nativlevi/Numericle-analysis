# neria atias 316118728
# nativ levi 209150879
# https://github.com/neriatias/Numerical-analysis


import numpy as np


def linear_interpolation(points, x):
    (x0, y0), (x1, y1) = points
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def polynomial_interpolation(points, x):
    (x0, y0), (x1, y1), (x2, y2) = points
    A = np.array([
        [x0 ** 2, x0, 1],
        [x1 ** 2, x1, 1],
        [x2 ** 2, x2, 1]
    ])
    B = np.array([y0, y1, y2])
    coef = np.linalg.solve(A, B)
    return coef[0] * x ** 2 + coef[1] * x + coef[2]


def lagrange_interpolation(points, x):
    (x0, y0), (x1, y1), (x2, y2) = points
    L0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
    L1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
    L2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))
    return y0 * L0 + y1 * L1 + y2 * L2


def main():
    points = []
    n = int(input("Enter the number of points (2 for linear, 3 for polynomial/lagrange): "))
    for i in range(n):
        x = float(input(f"Enter x{i}: "))
        y = float(input(f"Enter y{i}: "))
        points.append((x, y))

    x = float(input("Enter the x value for which you want to find the y value: "))

    print("Choose interpolation method:")
    print("1. Linear")
    print("2. Polynomial")
    print("3. Lagrange")
    choice = int(input("Enter your choice (1, 2, or 3): "))

    if choice == 1 and len(points) == 2:
        y = linear_interpolation(points, x)
    elif choice == 2 and len(points) == 3:
        y = polynomial_interpolation(points, x)
    elif choice == 3 and len(points) == 3:
        y = lagrange_interpolation(points, x)
    else:
        print("Invalid choice.")
        return

    print(f"The estimated y value at x = {x} is {y}")


if __name__ == "__main__":
    main()
