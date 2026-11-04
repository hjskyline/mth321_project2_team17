# Problem-specific derivation and stability argument (AI-assisted draft)

This is an argument for the prescribed linear smooth square benchmark, not an
instructor-approved proof. Team members must independently check it before using it.
Let gamma=0 or 2, h=1/N and s=q or r. Derivations of the two boundary rows are in
model.md. Here we justify the matrix structure and explain the different global rates.

## 1. Continuous uniqueness

The difference w of two solutions has zero Dirichlet data on the three prescribed
edges, satisfies -Delta w=0 and partial_n w+gamma*w=0 on the right. Integration by
parts gives integral_Omega |grad w|^2 + integral_right gamma*w^2 = 0. Both terms are
nonnegative. Thus w is constant, and its zero Dirichlet trace makes that constant zero.
This excludes a pure-Neumann gauge argument, which is unnecessary here.

## 2. Discrete comparison principle and stability

After eliminating the prescribed nodes, A has positive diagonal, nonpositive
off-diagonal entries and nonnegative row sums. The rows adjacent to eliminated
Dirichlet nodes include strictly positive row sums. The unknown-node adjacency graph
is connected in both directions, including the right-to-inward links for both closures.
The resulting irreducibly diagonally dominant Z-matrix is a nonsingular M-matrix,
so A^{-1} is componentwise nonnegative. Equivalently, the elementary discrete
minimum argument propagates any negative minimum through equal neighbours until
it reaches a strictly anchored row, a contradiction when Av>=0.

This supplies the comparison implication A a <= A b => a <= b.
It does not make the original matrix symmetric. Scaling a row changes raw residuals,
so the implementation separately reports scaled and componentwise backward errors.

Use v(x)=2x-x^2/2 sampled at the unknown nodes. It is nonnegative, <=3/2, and
A v >= 1 for each prescribed matrix. On strict interior nodes -Delta_h v=1;
eliminating its nonnegative Dirichlet traces only adds nonnegative contributions.
At the first-order right edge its derivative row gives 1+h/2+3gamma/2>=1.
At the ghost right edge its PDE row gives 1+2/h+3gamma/h>=1, with possible additional
nonnegative known-corner terms. Consequently A^{-1}1 <= v and

    ||A^{-1}||_infinity <= 3/2

for the original, unscaled mixed-row system. This is a mathematical bound specific
to these rows; it is NOT inferred from 1/minimum absolute row sum. Mixed row units
and scaling must still be respected when interpreting individual residual components.

## 3. Local residual does not have the same order on every row

Assume a smooth extension near x=1 with bounded derivatives independent of h.
Define tau=b-A u*, with exact boundary values eliminated in the same convention.
Taylor expansion gives |tau_int|<=C_i h^2. The one-sided derivative defect is O(h).
For the centred boundary condition,

    u(1+h)-u(1-h)=2h u_x(1)+(h^3/3)u_xxx(1)+O(h^5).

Replacing the true outside value by the ghost relation changes it by O(h^3);
dividing its contribution by h^2 gives a boundary PDE residual O(h). Therefore
|tau_right|<=C_b h for both closures, with constants depending on smooth derivatives.
The global second-order ghost claim cannot follow from falsely calling every row O(h^2).

## 4. Boundary-aware barriers explain the global orders

Let e=U-u* at unknown nodes. Then A e=tau and the exact Dirichlet error is zero.
Use z(x)=x sampled at the same nodes. Interior rows have A z>=0 after elimination.
On the right derivative rows A z>=1; on ghost PDE rows A z>=2/h (gamma>=0).

For first_order choose Phi=C_i h^2 v+C_b h z. Then A Phi>=|tau| on every row.
Since A^{-1}>=0, -Phi<=e<=Phi, yielding

    ||e||_infinity <= (3/2) C_i h^2 + C_b h = O(h).

For ghost choose Phi=C_i h^2 v+(C_b/2)h^2 z. At the right edge its second term
produces at least C_b h because A z>=2/h. The same comparison yields

    ||e||_infinity <= [(3/2) C_i+C_b/2] h^2 = O(h^2).

The constants assume the stated regularity and fixed nonnegative gamma. Our global
trapezoidal area weights sum to one, so the corresponding global discrete L2 is <=
the nodal maximum norm and obeys the same upper-order bounds. These are upper
bounds, not a promise that every special polynomial must show a nonzero leading term.

## 5. Weighted symmetry, corroboration and solver choice

Let W have interior diagonal weights 1. At right unknowns choose W=1/h for first_order
and W=1/2 for ghost. Then B=WA is symmetric: each inward coupling agrees with the
interior-to-boundary coefficient -1/h^2. Ghost tangential boundary links get half
weight; first-order boundary rows have no tangential links.

Writing c_kl=-B_kl>=0 and d_k=sum_l B_kl>=0 gives

    t^T B t = sum_{k<l} c_kl (t_k-t_l)^2 + sum_k d_k t_k^2.

Connectivity and at least one Dirichlet anchor force positivity for nonzero t, hence
B is positive definite. The program checks these structural assumptions and small
eigenvalues, but numerical eigenvalue samples alone are not the proof.

Production comparisons use GMRES on a shared diagonal row scaling of the ORIGINAL
nonsymmetric matrices. We do not add CG here: applying CG also requires its chosen
preconditioner to meet the necessary conditions, which an arbitrary AMG call need
not satisfy. Weighted symmetry is mathematical explanation, not an automatic claim
of a new algorithm or measured speedup.
