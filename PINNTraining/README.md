# PINN regressions

This folder contains files for PINN regressions.

There are three flow cases:

- steady 2D NACA0012 flow at Re=500,
- unsteady 2D NACA0012 flow at Re=500,
- unsteady 2D cylinder flow at Re=150.

and additional easy pde check case.

For each flow case and the additional case, there is a folder which contains codes for all PINN regressions done for the specific case (steady/unsteady NACA/ cylinder). The data folder should contain the simulation data - output from data transformation scripts in the Nektar++ simulations part.

Each code outputs a folder with results of the regressions along with a .mat file with predicted fields that can be plotted separately.

equations.py contains pde definitions
utitlities.py contains additional functions used by the scripts
