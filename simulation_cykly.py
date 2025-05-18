import numpy as np


def simulation_step(
        u: np.ndarray,
        lam: np.ndarray,
        rho_c: np.ndarray,
        dx: float,
        dy: float,
        dt: float) -> np.ndarray:
    """
    Provede jeden krok simulace pomocí výpočtového cyklu.

    Parametry:
        u      -- matice teplot (2D pole)
        lam    -- součinitel tepelné vodivosti
        rho_c  -- hustota * měrná tepelná kapacita
        dx, dy -- velikost buněk v prostoru
        dt     -- délka časového kroku

    Vrací:
        Nová matice teplot po jednom kroku
    """

    ny, nx = u.shape
    u_new = np.copy(u)

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            u_ij = u[i, j]
            lam_ij = lam[i, j]
            rho_c_ij = rho_c[i, j]

            # Levý soused
            lam_left = lam[i, j - 1]
            R_left = (1 / lam_ij + 1 / lam_left) * dx**2
            q_left = (u[i, j - 1] - u_ij) / R_left

            # Pravý soused
            lam_right = lam[i, j + 1]
            R_right = (1 / lam_ij + 1 / lam_right) * dx**2
            q_right = (u[i, j + 1] - u_ij) / R_right

            # Horní soused
            lam_up = lam[i - 1, j]
            R_up = (1 / lam_ij + 1 / lam_up) * dy**2
            q_up = (u[i - 1, j] - u_ij) / R_up

            # Dolní soused
            lam_down = lam[i + 1, j]
            R_down = (1 / lam_ij + 1 / lam_down) * dy**2
            q_down = (u[i + 1, j] - u_ij) / R_down

            delta_u = (2 / rho_c_ij) * dt * (q_left + q_right + q_up + q_down)

            u_new[i, j] = u_ij + delta_u

    return u_new
