import scipy.sparse as sparse
import numpy as np


def build_A_matrix(
        lambda_: np.ndarray,
        rho_c: np.ndarray,
        dx: float,
        dy: float,
        dt: float
        ) -> np.ndarray:
    """
    Sestaví řídkou matici A pro simulaci tepla pomocí maticového násobení.

    Parametry:
        lambda_ -- součinitel tepelné vodivosti (2D matice)
        rho_c   -- hustota * měrná tepelná kapacita
        dx, dy  -- velikosti buněk
        dt      -- časový krok

    Vrací:
        Matice A ve sparse formátu (CSR)
    """
    rows, cols = lambda_.shape
    size = rows * cols
    A = sparse.lil_matrix((size, size))

    def idx(i, j):
        return i * cols + j

    for i in range(rows):
        for j in range(cols):
            index = idx(i, j)
            lam = lambda_[i, j]
            rc = rho_c[i, j]

            coeff_x = 2 * dt / (rc * dx * dx)
            coeff_y = 2 * dt / (rc * dy * dy)

            diag = 0.0  # centrální prvek

            # Levý soused
            if j > 0:
                lam_l = lambda_[i, j - 1]
                w = coeff_x / (1 / lam + 1 / lam_l)
                A[index, idx(i, j - 1)] = w
                diag -= w

            # Pravý soused
            if j < cols - 1:
                lam_r = lambda_[i, j + 1]
                w = coeff_x / (1 / lam + 1 / lam_r)
                A[index, idx(i, j + 1)] = w
                diag -= w

            # Horní soused
            if i > 0:
                lam_u = lambda_[i - 1, j]
                w = coeff_y / (1 / lam + 1 / lam_u)
                A[index, idx(i - 1, j)] = w
                diag -= w

            # Dolní soused
            if i < rows - 1:
                lam_d = lambda_[i + 1, j]
                w = coeff_y / (1 / lam + 1 / lam_d)
                A[index, idx(i + 1, j)] = w
                diag -= w

            A[index, index] = diag

    return A.tocsr()
