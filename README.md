# rtms_optimize
Repo for building rTMS coil optimization method
*** 
# Mesh Correspondence between Surfaces 
***
## Construction of simNIBS optimized mesh geometry and comparison to fs_LR_32k

#### Objective:
Establish a correspondence between the individualized functional network parcellation embedded in **fs_LR_32k** space and the optimized mesh generated by the simNIBS re-meshing algorithm.  

#### Requirements:
Since we're modelling the grey matter and the volumetric mesh contains representations for different tissue types, we don't just want a surface projection but a **surface to discrete volume projection**, since this will better model the regions that we want to stimulate since it will have finite volume. 

# Progress Notes:

## Cross-registration of simNIBS -->fs_LR_32k

#### Components to work with:
1. simNIBS surface sphere with sulcal depth topology
2. fs_LR_32k standardized surface. 

Perform registration to MNINonLinear/Native using MSM. Then should be able to re-sample the individualized parcellations over, or alternatively should try to re-sample the sulcal surface onto fs_LR_32k. 

#### Steps to implement
- [x] Convert freesurfer spherical surface into GIFTI format
- [x] Convert freesurfer sulcal depth map to GIFTI metric map (.shape.gii)
- [x] Generate sphere.reg from Native to fsaverage
- [x] Generate registration sphere from Native --> fsaverage --> fs_LR_164k using -surface-sphere-project-unproject
- [x] Make MSMSulc source sphere
  - [x] Assign structure
  - [x] Convert height --> depth map
  - [x] Apply alignment transformation of native to fsaverage_LR_164k
  - [x] Apply affine registration
  - [x] Set radius at 100
- [x] Perform MSMSulc registration between mesh-generated sphere --> 164k MNINonLinear Native (or perhaps a lower resolution fs_LR_32k)
- [x] Resample individualized parcellation from MNINonLinear/fs_LR_32k --> optimized mesh (~68k)
- [ ] Project resampled labels to tetrahedral mesh generated by simNIBS using shared coordinate space between freesurfer and simNIBS

Since we'll be moving from surface mesh --> tetrahedral volumetric mesh. We'll use the surface points generated from Freesurfer as the correspondence points since the volumetric conversion of a mesh should not alter the surface vertices. We'll then fill tetrahedra with parcellation values if and only if the associated tetrahedra is a grey-matter component.

#### Additional Issues to address:
- [x] Upgrade from MSM_HOCR_v2 --> MSM_HOCR_v3 using source repo [MSM_HOCR](https://github.com/ecr05/MSM_HOCR)
- [x] Implement SVD removal of scaling/translational components after computing linear warp from native to fsaverage

## Bayesian Optimization of an rTMS Field Distribution-derived Objective Function

We're using a Bayesian Optimization approach here since evaluating the objective function (simulating the field distribution for a given position of the coil) is expensive. In addition we don't have direct access to the gradient information since we'll be working on a non-linear cost function with a constrained parameter surface (smoothed surface projected orthogonally from participant's head) so a sampling approach is called for. Using Bayesian Optimization with Gaussian Process priors assumes that our underlying objective function is continuous and smooth over our position/orientation parameters - this is not unreasonable. 

Toolbox to be used: [fmfn/BayesianOptimization](https://github.com/fmfn/BayesianOptimization).

##### Consideration: Parallelization of sampling and update approach? 

[Cornell-MOE](https://github.com/wujian16/Cornell-MOE) and the associated paper: [Parallel Bayesian Global Optimization of Expensive Functions
](https://arxiv.org/abs/1602.05149)

[Asynchronous Parallel Bayesian Optimisation via Thompson Sampling](https://arxiv.org/abs/1705.09236)


