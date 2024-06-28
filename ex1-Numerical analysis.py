'''מגישים:
נתיב לוי 209150879
נריה אטיאס 316118728
'''

import numpy as np


def is_invertible(matrix):
    return np.linalg.det(matrix) != 0


def inverse_using_elementary(matrix, integer_only=False):
    if not is_invertible(matrix):
        raise ValueError("המטריצה אינה הפיכה")

    n = matrix.shape[0]
    # יצירת מטריצת היחידה
    I = np.eye(n)
    # חיבור המטריצה המקורית עם מטריצת היחידה
    augmented_matrix = np.hstack((matrix, I))

    # ביצוע הפיכת מטריצות אלמנטריות
    for i in range(n):
        # חלוקה של השורה הנוכחית כדי להבטיח שהאלכסון הראשי הוא 1
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]

        # אפס את כל האלמנטים בעמודה הנוכחית, מלבד האלכסון הראשי
        for j in range(n):
            if i != j:
                augmented_matrix[j] = augmented_matrix[j] - augmented_matrix[j, i] * augmented_matrix[i]

    # המטריצה ההפוכה היא החלק הימני של המטריצה המורחבת
    inverse_matrix = augmented_matrix[:, n:]

    if integer_only:
        inverse_matrix = np.round(inverse_matrix).astype(int)

    return inverse_matrix


def max_row_norm(matrix):
    # מחשב את סכום הערכים המוחלטים בכל שורה
    row_sums = np.sum(np.abs(matrix), axis=1)
    # מוצא את הסכום המקסימלי מבין הסכומים
    max_norm = np.max(row_sums)
    return max_norm


# דוגמה למטריצה בגודל 3 על 3
A = np.array([
    [1, -1, -2],
    [2, -3, -5],
    [-1, 3, 5]
])

# חישוב נורמת השורה המקסימלית למטריצה המקורית
original_max_norm = max_row_norm(A)
print("נורמת השורה המקסימלית של המטריצה המקורית:", original_max_norm)

# חישוב המטריצה ההפוכה ונורמת השורה המקסימלית שלה
try:
    inverse_matrix = inverse_using_elementary(A, integer_only=True)
    print("המטריצה ההפוכה:")
    print(inverse_matrix)

    inverse_max_norm = max_row_norm(inverse_matrix)
    print("נורמת השורה המקסימלית של המטריצה ההפוכה:", inverse_max_norm)

    # חישוב המכפלה בין הנורמות ושמירתה במשתנה COND
    COND = original_max_norm * inverse_max_norm
    print("הערך של COND:", COND)
except ValueError as e:
    print(e)
