import numpy as np
import galois
GF2 = galois.GF(2)

def inverse_GF2(A):
    return np.linalg.inv(GF2(A))

# https://npdeep.github.io/matrix-inversion-gf2.html
def RREF_binary(A):
    """Converts a matrix to reduced row echelon form (RREF)"""
    n_rows, n_cols = A.shape

    # Compute row echelon form (REF)
    current_row = 0
    for j in range(n_cols):  # For each column
        if current_row >= n_rows:
            break

        pivot_row = current_row

        # find the first row in this column with non-zero entry.
        # this becomes the pivot row
        while pivot_row < n_rows and A[pivot_row, j] == 0:
            pivot_row += 1

        # if we reach the end, this column cannot be eliminated.
        if pivot_row == n_rows:
            continue

        # otherwise, swap current row with the pivot row
        A[[current_row, pivot_row]] = A[[pivot_row, current_row]]

        pivot_row = current_row
        current_row += 1

        # Eliminate rows below
        for i in range(current_row, n_rows):
            # subtract current row from any other rows beneath with
            # a non-zero entry in this current column
            if A[i, j] == 1:
                A[i] = (A[i] +  A[pivot_row]) % 2 # subtracting is same as adding in GF(2)


    # Compute reduced row echelon form (RREF)
    # in the RREF form, there is only one non-zero entry in a column.
    for i in reversed(range(current_row)):
        # Find pivot
        pivot_col = 0

        # find the column with the first non-zero entry.
        while pivot_col < n_cols and A[i, pivot_col]==0:
            pivot_col += 1
        if pivot_col == n_cols:
            continue  # Skip this all-zero row

        # Eliminate this column in all the rows above
        for j in range(i):
            if A[j, pivot_col] == 1:
                A[j] = (A[j] +  A[i]) % 2

    return A


def inverse_binary(A):
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        raise Exception("Matrix has to be square")

    augmented_matrix = np.hstack([A, np.eye(n_rows)]) # Augmented matrix
    rref_form = RREF_binary(augmented_matrix)
    # return the second half of the augmented matrix
    return rref_form[:, n_rows:]


def RREF_binary_(A):
    """Converts a matrix to reduced row echelon form (RREF)"""
    n_rows, n_cols = A.shape
    A = A.astype(bool)

    # Compute row echelon form (REF)
    current_row = 0
    for j in range(n_cols):  # For each column
        if current_row >= n_rows:
            break

        pivot_row = current_row

        # find the first row in this column with non-zero entry.
        # this becomes the pivot row
        while pivot_row < n_rows and A[pivot_row, j] == 0:
            pivot_row += 1

        # if we reach the end, this column cannot be eliminated.
        if pivot_row == n_rows:
            continue

        # otherwise, swap current row with the pivot row
        A[[current_row, pivot_row]] = A[[pivot_row, current_row]]

        pivot_row = current_row
        current_row += 1

        tmp_range = np.arange(current_row, n_rows)
        tmp_range = tmp_range[A[tmp_range, j]]
        A[tmp_range, :] ^= A[pivot_row, :]

    # Compute reduced row echelon form (RREF)
    # in the RREF form, there is only one non-zero entry in a column.
    for i in reversed(range(current_row)):
        # Find pivot
        tmp = A[i, :].nonzero()[0]

        if len(tmp) == 0:
            continue  # Skip this all-zero row

        # find the column with the first non-zero entry.
        pivot_col = tmp[0]

        tmp_range = np.arange(i)
        tmp_range = tmp_range[A[tmp_range, pivot_col]]
        A[tmp_range, :] ^= A[i, :]

    return A


def inverse_binary_(A):
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        raise Exception("Matrix has to be square")

    augmented_matrix = np.hstack([A, np.eye(n_rows, dtype=bool)]) # Augmented matrix
    rref_form = RREF_binary(augmented_matrix)
    # return the second half of the augmented matrix
    return rref_form[:, n_rows:]