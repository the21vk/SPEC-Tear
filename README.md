# **SP**ectral **E**igenvalue **C**ode for the **T**earing instability

Eigenvalue code to obtain the growth rates for the 3D tearing instability using spectral methods.



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Overview

The code calculates growth rates for MHD instabilities by solving the linearized MHD equations in a periodic domain. It uses spectral methods with Fast Fourier Transforms (FFT) for accurate computation of spatial derivatives and constructs the system matrices efficiently.

Key features:
- Spectral analysis of MHD instabilities
- Computation of growth rates across parameter space
- Visualization of dispersion relations
- Analysis of stability boundaries

## Files

- `spectear.py` - Core computational functions for MHD stability analysis
- `exampleUsage.ipynb` - Jupyter notebook for computing and visualizing dispersion relations

## Requirements

- Python 3.7+
- NumPy
- SciPy
- PyTorch
- Matplotlib
- Jupyter (for notebooks)

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/SPEC-Tear.git
cd SPEC-Tear
```

Install dependencies:
```bash
pip install numpy scipy torch matplotlib jupyter
```

## Usage

### Using the Python Module

```python
from mhd_stability import compute_max_real_eigenvalue

# Calculate growth rate for specific parameters
growth_rate = compute_max_real_eigenvalue(
    k_star=0.7,  # Wavenumber
    lam=2.8,     # Width parameter
    N=42,        # Grid resolution
    S=100        # Lundquist number
)

print(f"Growth rate: {growth_rate}")
```


## Theory Background

The code calculates the growth rate for the tearing instability in 3D by computing the eigenvalues of the linearized MHD equations as presented in [our work](https://arxiv.org/abs/2412.10065v1). A positive real part in any eigenvalue indicates instability.

Key parameters:
- `k_star`: Wavenumber
- `lam`: Width parameter controlling the profile function shape
- `S`: Lundquist number (ratio of resistive to Alfvén time scales)
- `N`: Grid resolution for the spectral method

Example usage is presented in the notebook [here](exampleUsage.ipynb). If required, one can obtain the eigenvectors as well by modifying the code slightly. See [PyTorch documentation](https://pytorch.org/docs/stable/index.html) to learn how to access the eigenvectors. 


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{spec-tear,
  author = {Your Name},
  title = {SPEC-Tear: Spectral Eigenvalue Code for the Tearing Instability},
  year = {2025},
  url = {https://github.com/yourusername/mhd-stability}
}
```

## Future Plans

- Add support for parsing linearized equations as strings, constructing relevant derivative operators, and then solving for the eigenvalues and eigenvectors.

## Acknowledgements

- Discussions with Prayush Kumar, Saumav Kapoor and Vaishak Prasad at ICTS were extremely useful.
- Claude.ai generated the comments that describe the code line by line.
