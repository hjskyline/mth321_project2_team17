# Direction 2 baseline: model and discretisation

## Continuous problem and manufactured data

Solve `-Delta u = f` on the unit square `(0,1)^2`.
The left, bottom and top edges, including all four corners, carry prescribed
Dirichlet values `g`. The open right edge carries either Neumann or Robin data.
The benchmark is manufactured; it is not a calibrated engineering experiment.

- Exact solution: `u*(x,y) = exp(x+y)`.
- Source: `f = -u*xx-u*yy = -2 exp(x+y)`.
- Dirichlet trace: `g = u*` on the three Dirichlet edges.
- On `x=1`, the outward normal is `(1,0)`, so `partial_n u = u_x`.
- Neumann data: `q(y) = exp(1+y)`.
- Robin data: `u_x + beta*u = r`, with `beta=2` and `r(y)=3 exp(1+y)`.

The solution is unique for this mixed problem with a nonempty Dirichlet part.
There is no pure-Neumann compatibility or constant-gauge problem here.
The exact solution is positive, increases in both coordinates, has no assumed
reflection symmetry, and ranges from 1 to exp(2) on the closed square.

## Mesh and unknowns

`N` is the number of intervals per coordinate, `h=1/N`, and
`(x_i,y_j)=(i*h,j*h)` for `i,j=0,...,N`.
Unlike the old Dirichlet-only scaffold, right-edge values are unknown.

Store `i=1,...,N` and `j=1,...,N-1`, in the order

`k(i,j) = (i-1)*(N-1) + (j-1)`.

There are `N*(N-1)` unknowns. The left edge and the bottom/top edges are
eliminated; the two right corners remain Dirichlet, not Neumann/Robin unknowns.
`DiscreteSystem.full_solution()` restores the complete `(N+1,N+1)` nodal grid.

## Interior equation

At `1 <= i < N`, `1 <= j < N`:

`(4 U_ij - U_(i-1,j) - U_(i+1,j) - U_(i,j-1) - U_(i,j+1))/h^2 = f_ij`.

A known neighbour with coefficient `a` contributes `-a*g` to the right-hand
side. In particular, an interior Dirichlet neighbour contributes `+g/h^2`.

## Two prescribed right-edge closures

Write the boundary condition as `u_x + gamma*u = s`, where:

| Boundary | gamma | s(y) |
| --- | --- | --- |
| Neumann | 0 | q(y) |
| Robin | beta=2 | r(y) |

### One-sided derivative row (`first_order`)

Approximate `u_x` by `(U_Nj-U_(N-1,j))/h`:

`(1/h+gamma)*U_Nj - U_(N-1,j)/h = s_j`.

This derivative equation REPLACES the PDE at the right boundary.

### Ghost-point elimination (`ghost`)

Introduce a conceptual outside value `U_(N+1,j)` and use

`(U_(N+1,j)-U_(N-1,j))/(2h) + gamma*U_Nj = s_j`.

Hence `U_(N+1,j) = U_(N-1,j) + 2h*(s_j-gamma*U_Nj)`.
Substitution into the five-point PDE at `(1,y_j)` gives

`(4/h^2+2*gamma/h)*U_Nj - 2*U_(N-1,j)/h^2`
` - U_(N,j-1)/h^2 - U_(N,j+1)/h^2 = f_Nj + 2*s_j/h`.

No outside value is numbered or stored. At `j=1` or `N-1`, the known corner
term is moved to the right-hand side. The doubled inward coefficient is deliberate.

The two closures change the boundary formulation as well as the derivative
formula. A centred derivative alone does not prove second-order global error;
the eliminated ghost PDE row can have first-order local truncation, while the
smooth mixed problem can nevertheless show second-order global convergence.

## A hand-checkable boundary row

Use `N=4`, `h=1/4`, right-edge point `(i,j)=(4,1)`, so `k=9`, inward `k=6`.
Set `t=exp(1.25)`.

- One-sided: `A[9,9]=4+gamma`, `A[9,6]=-4`, `b[9]=(1+gamma)*t`.
- Ghost: `A[9,9]=64+8*gamma`, `A[9,6]=-32`, `A[9,10]=-16`,
  `b[9]=-2*t + 8*(1+gamma)*t + 16*exp(1)`.
- The last RHS term is the bottom-right Dirichlet corner contribution.

The tests compare these explicit small-grid values with the assembled row.

## Solver and residuals

Build in COO and convert once to CSR. The matrices are generally nonsymmetric:
one-sided rows use different coupling/scaling; ghost elimination doubles the
inward coupling. Use sparse direct LU (`spsolve`), not CG. It has no user tolerance.

For the residual vector `res=b-A*U`, where `*` here denotes mathematical matrix
multiplication (Python uses `@`), report:

- Raw infinity norm: `max(abs(res))`.
- Diagonally row-scaled residual: `max_i abs(res_i)/abs(A_ii)`.
- Componentwise backward error:
  `max_i abs(res_i)/(abs(A)@abs(U)+abs(b))_i`.

The latter two are unchanged by positive rescaling of any equation, important
when derivative rows scale like `h^-1` and PDE rows like `h^-2`.
The reported asymmetry uses Frobenius norms: `||A-A.T||_F / ||A||_F`.

An independent BiCGSTAB solve on a diagonally row-scaled system uses `rtol=1e-12`.
On each finest mesh, its difference from LU must be below `1e-3` times the
measured MMS error in BOTH global norms. This supports algebraic-error
separation for this benchmark; shared assembly is not an independent discretisation
check, and solver agreement is not a rigorous exact forward-error bound.

## Error norms and predictions recorded before the repository baseline run

Let `e_ij=U_ij-u*(x_i,y_j)`. Dirichlet errors are zero by construction.

- Global discrete L2: `sqrt(h^2 sum_ij w_i w_j e_ij^2)`, where endpoints have
  weight `w=1/2` and other nodes have weight 1. Right-edge unknowns therefore
  receive half an area weight.
- Global nodal Linf: `max_ij abs(e_ij)`.
- Interior L2/Linf: restrict to `i,j=1,...,N-1`; L2 uses `h^2` area weights.
- Right-edge L2/Linf: restrict to `i=N, j=1,...,N-1`; L2 uses `h` length weights.
  A boundary L2 norm has a different dimensional scaling from a domain L2 norm.

Analytical/pack-based predictions (not fitted output):

1. Source is negative and outward right derivative is positive; exponential
   data are not reflection symmetric.
2. An affine manufactured solution should be reproduced to solver precision
   by all four combinations. A quadratic should be nodally exact for the ghost
   scheme, but is not a valid nontrivial convergence benchmark for that scheme.
3. For the smooth exponential benchmark, one-sided boundary Linf should
   approach order 1; ghost global L2/Linf should approach order 2.
4. Matrix diagonal entries are positive, off-diagonal entries nonpositive,
   and the matrix is generally nonsymmetric.
5. Initial meshes are `N=8,16,32,64`. Fits on the last three levels are diagnostic;
   check per-step orders and fit deviations before selecting a report window.

Runtime outputs are in results/ and figures/. Do not overwrite these predictions
to match observations. This foundation does not yet include pure Neumann data,
real heat-transfer calibration, a solver-performance ranking, or a convergence proof.
