'''מגישים:
נתיב לוי 209150879
נריה אטיאס 316118728

https://github.com/nativlevi/Numericle-analysis.git
'''

import numpy as np

# פונקציה לבדיקת הפיכות המטריצה על ידי חישוב הדטרמיננטה שלה
def is_invertible(matrix):
    return np.linalg.det(matrix) != 0

# פונקציה למציאת המטריצה ההפוכה באמצעות מטריצות אלמנטריות
def inverse_using_elementary(matrix, integer_only=False):
    if not is_invertible(matrix):
        raise ValueError("המטריצה אינה הפיכה")

    n = matrix.shape[0]
    I = np.eye(n)  # יצירת מטריצת היחידה
    augmented_matrix = np.hstack((matrix, I))  # חיבור המטריצה המקורית עם מטריצת היחידה

    for i in range(n):
        diag_element = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / diag_element  # נרמול האלכסון הראשי

        for j in range(n):
            if i != j:
                row_factor = augmented_matrix[j, i]
                augmented_matrix[j] = augmented_matrix[j] - row_factor * augmented_matrix[i]  # איפוס הערכים בעמודות אחרות

    inverse_matrix = augmented_matrix[:, n:]  # המטריצה ההפוכה היא החלק הימני של המטריצה המורחבת

    if integer_only:
        inverse_matrix = np.round(inverse_matrix).astype(int)

    return inverse_matrix

# פונקציה לפירוק LU של מטריצה
def lu_decomposition(matrix):
    n = matrix.shape[0]
    L = np.eye(n)  # יצירת מטריצת היחידה L
    U = matrix.copy()  # יצירת עותק של המטריצה המקורית למטריצת U

    for i in range(n):
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor  # עדכון מטריצת L
            U[j] = U[j] - factor * U[i]  # עדכון מטריצת U

    return L, U

# פונקציה לפתרון מערכת המשוואות LY = b ו-UX = Y
def solve_lu(L, U, b):
    n = L.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)

    # פתרון LY = b
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # פתרון UX = Y
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

# הדוגמא למטריצה בגודל 3x3
A = np.array([
    [1, 4, -3],
    [-2, 1, 5],
    [3, 2, 1]
])

# וקטור פתרון לדוגמה
b = np.array([1,2,3])

try:
    # חישוב המטריצה ההפוכה
    inverse_matrix = inverse_using_elementary(A, integer_only=False)
    print("המטריצה ההפוכה:")
    print(inverse_matrix)

    # ביצוע פירוק LU
    L, U = lu_decomposition(A)
    print("מטריצת L:")
    print(L)
    print("מטריצת U:")
    print(U)

    # פתרון מערכת המשוואות A*x = b
    x = solve_lu(L, U, b)
    print("פתרון מערכת המשוואות:")
    print(x)
except ValueError as e:
    print(e)
