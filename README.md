# rtms_optimize
Repo for building rTMS coil optimization method
*** 
# Mesh Correspondence between Surfaces 
***

## TO-DO:

- [x] Develop framework surrounding handling of file IO of simulation files and scoring
- [ ] Wrap surface parameterization of local patches into geolib library function (extract_surface_patch notebook)
- [x] Write up Bayesian Optimization component within a module (rtms_optimize notebook)
- [ ] Implement extensive unit testing of geolib notebook
- [ ] Look into tetrapro's weird behaviour generating values that are very slightly greater than 1 (might be due to resampling)


# Progress Notes:

## Cross-registration of simNIBS -->fs_LR_32k

TO-DO: Update with new post-Freesurfer scripts

## Ribbon-Constrained tetrahedral projection algorithm
Analagous to HCP surface projection algorithm. Use Monte Carlo sampling with uniform sampling within tetrahedral volumes. Uniform sampling is achieved through the parallelepiped folding algorithm: [Generating Random Points in a Tetrahedron](http://vcg.isti.cnr.it/publications/papers/rndtetra_a.pdf) by C.Rocchini and P. Cignoni. 

#### Implementation Steps:
1. Optimize loop run-time using Numba static typing
2. Use variance convergence criterion of Monte Carlo sampling so number of samples is chosen dynamically
3. Parallelize across loops (using no-gil of Numba???)
4. Wrap in a Python script for easier deployment

## Bayesian Optimization of an rTMS Field Distribution-derived Objective Function

We're using a Bayesian Optimization approach here since evaluating the objective function (simulating the field distribution for a given position of the coil) is expensive. In addition we don't have direct access to the gradient information since we'll be working on a non-linear cost function with a constrained parameter surface (smoothed surface projected orthogonally from participant's head) so a sampling approach is called for. Using Bayesian Optimization with Gaussian Process priors assumes that our underlying objective function is continuous and smooth over our position/orientation parameters - this is not unreasonable. 

Toolbox to be used: [fmfn/BayesianOptimization](https://github.com/fmfn/BayesianOptimization).

##### Consideration: Parallelization of sampling and update approach? 

[Cornell-MOE](https://github.com/wujian16/Cornell-MOE) and the associated paper: [Parallel Bayesian Global Optimization of Expensive Functions
](https://arxiv.org/abs/1602.05149)

[Asynchronous Parallel Bayesian Optimisation via Thompson Sampling](https://arxiv.org/abs/1705.09236)

## Debug Notes

1. When installing Cornell-MOE, update setup.py as per instructions in optimization/README.md.
2. When constructing FEM, use Gmsh4 (MSH4), then convert into MSH2. The gmsh-sdk will be able to handle both. If you use SimNIBS' Gmsh3 to construct the FEM, the sdk will not be able to read the MSH3 file correctly.
3. Cornell-MOE works with numpy-1.16
