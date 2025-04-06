import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.sparse.linalg import LinearOperator
import torch

def compute_max_real_eigenvalue(k_star, lam, N=42, S=100):
    """
    Computes the maximum real part of eigenvalues for a magnetohydrodynamic (MHD) instability analysis.
    
    This function analyzes the stability of a plasma system by computing eigenvalues of the
    linear stability operator. It uses spectral methods with FFT for spatial derivatives
    and constructs the system matrices using LinearOperator for memory efficiency.
    
    Parameters:
    ----------
    k_star : float
        Wavenumber parameter for the analysis, representing spatial frequency.
    lam : float
        Width parameter controlling the shape of the profile function.
    N : int, optional
        Grid resolution (number of points in each dimension). Default is 42.
    S : float, optional
        Lundquist number, representing the ratio of resistive to Alfvén time scales. Default is 100.
        
    Returns:
    -------
    float
        The maximum real part of the eigenvalues, indicating system stability.
        Positive values indicate instability, with larger values corresponding to faster growth rates.
    """

    # Define the spatial grid and Fourier frequencies
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)  # Spatial grid in x direction
    q = fftfreq(N, x[1] - x[0]) * 2 * np.pi  # Corresponding wavenumbers

    # Define spatial derivative operators using FFT
    def DX(func):
        """Compute x-derivative using spectral method"""
        func_hat = fft2(func)  # Transform to Fourier space
        q_func_hat = 1j * q[:, None] * func_hat  # Multiply by wavenumber (derivative in Fourier space)
        return (ifft2(q_func_hat))  # Transform back to physical space

    def DY(func):
        """Compute y-derivative using spectral method"""
        func_hat = fft2(func)  # Transform to Fourier space
        q_func_hat = (1j * q[:, None] * func_hat.T).T  # Multiply by wavenumber (derivative in Fourier space)
        return (ifft2(q_func_hat))  # Transform back to physical space

    # Define the profile function 'f' with low-pass filtering
    X, Y = np.meshgrid(x, x, indexing='ij')  # 2D spatial grid
    # Initial profile function (represents equilibrium magnetic field or flow)
    f = 2.6 * np.tanh(X) * np.cosh(X) ** -2 * np.cosh(Y/lam) ** -2
    
    # Apply low-pass filter to remove high-frequency components (ensures periodicity of the equilibrium)
    f_hat = fft2(f)
    cutoff = 18  # Cutoff frequency for the low-pass filter
    low_pass_filter = np.zeros_like(f_hat)
    # Keep only low frequency components
    low_pass_filter[:cutoff, :cutoff] = 1
    low_pass_filter[-cutoff:, :cutoff] = 1
    low_pass_filter[:cutoff, -cutoff:] = 1
    low_pass_filter[-cutoff:, -cutoff:] = 1
    f_filtered_hat = f_hat * low_pass_filter
    f = (ifft2(f_filtered_hat))  # Filtered profile function

    # Define the right-hand side operator of the eigenvalue problem
    def RHS(state_ux_uy_bx_by):
        """
        Right-hand side operator.
        
        Parameters:
        ----------
        state_ux_uy_bx_by : ndarray
            Flattened state vector containing velocity (ux, uy) and 
            magnetic field perturbations (bx, by)
            
        Returns:
        -------
        ndarray
            Result of applying the operator to the state vector
        """
        # Split and reshape the state vector into its components
        ux, uy, bx, by = np.split(state_ux_uy_bx_by, 4)
        ux = np.reshape(ux, (N, N))
        uy = np.reshape(uy, (N, N))
        bx = np.reshape(bx, (N, N))
        by = np.reshape(by, (N, N))

        # Velocity equation x-component (linearized momentum equation)
        out_ux = (
            DY(DY(f)) * by
            + DX(DY(f)) * bx
            + DX(f) * DY(bx)
            - DY(f) * DX(bx)
            - f * DX(DY(bx))
            - f * DY(DY(by))
            + f * k_star**2 * by
        )
        
        # Velocity equation y-component
        out_uy = (
            -DX(DY(f)) * by
            - DY(f) * DX(by)
            - DX(DX(f)) * bx
            + DX(f) * DY(by)
            - f * k_star**2 * bx
            + f * DX(DX(bx) + DY(by))
        )
        
        # Magnetic field equation x-component (linearized induction equation)
        out_bx = 1j * k_star * f * ux + 1 / S * (DX(DX(bx)) + DY(DY(bx)) - k_star**2 * bx)
        
        # Magnetic field equation y-component
        out_by = 1j * k_star * f * uy + 1 / S * (DX(DX(by)) + DY(DY(by)) - k_star**2 * by)

        # Combine and return flattened result
        return np.hstack(
            (np.ravel(out_ux), np.ravel(out_uy), np.ravel(out_bx), np.ravel(out_by))
        )

    # Define the left-hand side operator of the eigenvalue problem
    def LHS(state_ux_uy_bx_by):
        """
        Left-hand side operator.
        
        Parameters:
        ----------
        state_ux_uy_bx_by : ndarray
            Flattened state vector
            
        Returns:
        -------
        ndarray
            Result of applying the operator to the state vector
        """
        # Split and reshape the state vector
        ux, uy, bx, by = np.split(state_ux_uy_bx_by, 4)
        ux = np.reshape(ux, (N, N))
        uy = np.reshape(uy, (N, N))
        bx = np.reshape(bx, (N, N))
        by = np.reshape(by, (N, N))

        # Apply the incompressibility constraint to velocity components
        out_ux = DY(-DX(ux) - DY(uy)) / 1j / k_star - 1j * k_star * uy
        out_uy = -DX(-DX(ux) - DY(uy)) / 1j / k_star + 1j * k_star * ux

        # Combine with magnetic field components (identity operation for these)
        return np.hstack(
            (np.ravel(out_ux), np.ravel(out_uy), np.ravel(bx), np.ravel(by))
        )

    # Create LinearOperators for memory efficiency
    size = 4 * N * N  # Total size of the system (4 variables × N² grid points)
    LHS_op = LinearOperator((size, size), matvec=LHS)
    RHS_op = LinearOperator((size, size), matvec=RHS)

    # Convert to dense matrices for eigenvalue computation
    LHS_mat = LHS_op @ np.eye(size)
    RHS_mat = RHS_op @ np.eye(size)
    print('Built operators')
    
    # Convert to PyTorch tensors for more efficient eigenvalue computation
    LHS_mat_tensor = torch.tensor(LHS_mat)
    RHS_mat_tensor = torch.tensor(RHS_mat)

    # Final matrix for eigenvalue problem: A⁻¹B
    Final_Matrix = torch.linalg.inv(LHS_mat_tensor) @ RHS_mat_tensor
    print('Inverted matrix')

    # Compute eigenvalues
    evals_pt, _ = torch.linalg.eig(Final_Matrix)
    evals = evals_pt.numpy()

    # Return the maximum real part of the eigenvalues (growth rate)
    return np.max(np.real(evals))


# Example usage: Calculate growth rates for specific parameters
if __name__ == "__main__":
    # Parameter values to test
    k_star_values = [0.7]    # Wavenumber
    lam_values = [2.8]       # Profile width parameter
    
    # Array to store results
    gam_arr = np.zeros((len(k_star_values), len(lam_values)))
    
    # Compute eigenvalues for each parameter combination
    for i, k in enumerate(k_star_values):
        for j, l in enumerate(lam_values):
            # Compute with reduced grid size (N=30) for speed
            result = compute_max_real_eigenvalue(k_star=k, lam=l, N=30, S=100)
            print(f'k_star = {k}, width = {l}, max real part = {result}')
            gam_arr[i, j] = result
    
    # Note: For a full parameter scan, you might want to save results to a file:
    # np.save('growth_rates.npy', gam_arr)