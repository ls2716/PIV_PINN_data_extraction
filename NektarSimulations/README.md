# Nektar simulations

This folder contains files for Nektar++ simulations.

There are three flow cases:

- steady 2D NACA0012 flow at Re=500,
- unsteady 2D NACA0012 flow at Re=500,
- unsteady 2D cylinder flow at Re=150.

For each flow case, there is a folder which contains:

- 6 folders with convergence cases with Nektar++ simulation definition file and mesh definition file, as well as otput file with forces vs iteration,
- convergence plot folder that contains the code for plotting the convergence plots from the forces data,
- data_analysis_transformation folder with code for output data analysis and transformation,
- the unsteady cases additionally have a rerun case to gain the mean flow data.

Additionally, the cylinder contains cyl.stp file which contains cylinder geometry definition used by Nekar++ mesher.

## Simulation procedure

1. Nektar++ simulations wre performed.
2. Convergence was plotted using plot.py file in convergence_plots folder.
3. (unsteady) cycle_analysis.py was used to find cycle length expressed in iterations.
4. (unsteady) Additional rerun was performed with mean data filter.
5. point_gen.py file was used to define points for field interpolation.
6. The field data was interpolated for the given points.
7. \*\_mat_gen.py file extracted field data from the interpolated file - this output is fed to PINN training.
8. (unsteady) forcing.py and forcing_alt.py files were used to obtain unsteady forcing data (used for results comparison).
