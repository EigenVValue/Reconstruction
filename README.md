In this exercise you will:

* Compute an implicit MLS function that approximates a cloud of 2D/3D points with normals. The input points will then lie at the zero level set of the computed function.

* Sample the implicit function on a three dimensional volumetric grid.

* Reconstruct a triangle mesh / contours of the zero level set of the implicit function, by using the Marching Cubes/Squares algorithm on the grid.

* Experiment with various MLS reconstruction parameters.


## Part 1: setting up constraints

__Task 1__: Implement the function `sample_constraints(...)` in `task1_constraints.py` that computes the constraints as described above. The function should return a list of tuples of the form $(p_i, f_i)$, where $p_i$ is a point in the input point cloud and $f_i$ is the corresponding constraint value.

Result visualizations for simple data (circle) should look like this:

Input points & normals|  Sampled constraints
:-------------------------:|:-------------------------:
<img src="data/1.1.png" width="250"> |  <img src="data/1.2.png" width="250"> 

## Part 2: MLS Interpolation

 __Task 2__: 
 Please implement the missing part of `global_predictor(...)` function in `task2_solver.py`. 

1) First, convert grid and constraint points into required polynomial basis (degree). You may use transform_to_polynomial_basis function for this

2) Then solve linear least squares problem. Optimizations problem:\
$X \cdot c = v$\
$`c = \text{argmin}_{c}  \sum_i  ||v_i - x_i \cdot c||_2^2`$
\
$c = ...$ - use analytical solution described in class (linear least squares) \
\
Where:
\
$X$ (matrix) - constraint point in polynomial basis (constr_pts) \
$x$ (vector) - constraint point in polynomial basis (constr_pts[i]) \
$v$ (vector) - the implicit function values in constraints points (constr_vals[i]) \
$c$ (vector) - coefficients of the polynomial (that approximates the shape)


3) Finally, interpolate implicit function in the grid points, e.g.:\
$g$ - a grid point (grid_pts[i])\
$\hat v$ - the predicted value at the grid point\
$\hat v = g \cdot c$
 

Result visualizations for simple data (circle) should look like this:

 Inference on a grid | Isosurface Reconstruction 
:-------------------------:|:-------------------------:
 <img src="data/1.3.png" width="250"> | <img src="data/1.4.png" width="250">

 ## Part 3: Local model

__Task 3.1__: \
Please implement part of  `local_predictor(...)` in `task2_solver.py`. You need to find nearest neighbours in a fixed radius

Given a grid point, it's nearest constrain points and constraint values, build polynomial (of specified degree) that approximates the constraint points in the local space. The optimization problem for each grid point is:
\
$`c = \text{argmin}_{c}  \sum_{x_i \in N_X(g)}  w_i * ||v_i - x_i \cdot c||_2^2`$\
$c = ...$ - use analytical solution described in class (weighted linear least squares)
\
\
$g$ - grid point in polynomial basis (grid_pts[i])\
$N_X(g)$ - nearest constraint points in the neighbourhood of grid point $g$  \
$x_i$ (vector) - constraint point in polynomial basis (constr_pts[i]) \
$v_i$ (vector) - the implicit function values in constraints points (constr_vals[i]) \
$c$ (vector) - coefficients of the polynomial (that approximates the shape)\
\
Note that we'll build __weighted__ least squares. This means that error at each of the constraint points will be weighted. The weight is the wendland function (is implemented in `task2_solver.py`). The smaller the distance between the grid and the constraint point is the bigger is the weight term for this point\
$w_i$ (scalar) - the weight of the constraint point. $w_i$ = wendland(distance($g$, $x_i$))



Finally, interpolate implicit function in the grid points, e.g.:\
$\hat v$ - the predicted value at the grid point $g$\
$\hat v = g \cdot c$

__Task 3.2__: Please implement part of  `eval_grid_point(...)` in `task2_solver.py`. 

The result of local model on a more complicated data should look like this:

 Sampled constrains | Inference on a grid | Isosurface Reconstruction 
:-------------------------:|:-------------------------:|:-------------------------:
 <img src="data/3.3.png" width="250"> | <img src="data/3.4.png" width="250"> | <img src="data/3.5.png" width="250">

The model approximates the data, but has some noise. Let's regularize the model.


__Task 3.3__:
Please add regularization for optimization in  `eval_grid_point(...)` in `task2_solver.py`. 

$` c = \text{argmin}_{c}  \sum_{x_i \in N_X(g)}  w_i * ||v_i - x_i \cdot c||_2^2  + |c|_2^2 `$

Final results:

 Inference on a grid | Isosurface Reconstruction 
|:-------------------------:|:-------------------------:
  <img src="data/3.4.png" width="250"> | <img src="data/3.7.png" width="250">