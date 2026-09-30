Yes. Using the definition in the **Hellinger-distance lecture** and the **GMM extension used by Mercurio & Singla**, the algorithm is naturally a **two-level computation**:

1. Compute the Hellinger distance between every pair of Gaussian components.
2. Solve a linear program to optimally associate the mixture weights between the two GMMs.

The lecture defines Hellinger distance through the square-root PDFs and relates its square to the Bhattacharyya/Fidelity coefficient.  Mercurio & Singla then use the closed-form Gaussian Hellinger distance and extend it to GMMs through a linear-programming formulation. 

## 1. Input GMMs

Let the two probability density functions be

$$
P(x)=\sum_{i=1}^{N}\alpha_i
\mathcal N(x;\mu_i^P,\Sigma_i^P)
$$

and

$$
Q(x)=\sum_{j=1}^{M}\beta_j
\mathcal N(x;\mu_j^Q,\Sigma_j^Q),
$$

where

$$
\sum_{i=1}^{N}\alpha_i=1,
\qquad
\sum_{j=1}^{M}\beta_j=1.
$$

This is the GMM formulation used in the Mercurio paper. 

---

# 2. Compute component-wise Hellinger distances

For each pair of components \(i,j\), define

$$
p_i(x)=\mathcal N(x;\mu_i^P,\Sigma_i^P)
$$

and

$$
q_j(x)=\mathcal N(x;\mu_j^Q,\Sigma_j^Q).
$$

The Hellinger distance between two arbitrary PDFs is

$$
D_H(p_i,q_j)
=
\sqrt{
1-
\int
\sqrt{p_i(x)q_j(x)}\,dx
}.
$$

The lecture gives the equivalent definition

$$
h^2(P,Q)=1-F(P,Q),
$$

where \(F\) is the Bhattacharyya/Fidelity coefficient. 

For Gaussian PDFs, Mercurio & Singla give the closed form

$$
D_H(p_i,q_j)
=
\sqrt{
1-
\frac{
|\Sigma_i^P|^{1/4}
|\Sigma_j^Q|^{1/4}
}{
|\Sigma_{ij}|^{1/2}
}
\exp\left[
-\frac12
(\mu_i^P-\mu_j^Q)^T
\Sigma_{ij}^{-1}
(\mu_i^P-\mu_j^Q)
\right]
},
$$

where

$$
\boxed{
\Sigma_{ij}
=
\frac12
\left(
\Sigma_i^P+\Sigma_j^Q
\right)
}
$$

as given in Eq. (15) of the paper. 

Therefore, construct the \(N\times M\) distance matrix

$$
\mathbf D =
\begin{bmatrix}
D_H(p_1,q_1)&\cdots&D_H(p_1,q_M)\\
\vdots&\ddots&\vdots\\
D_H(p_N,q_1)&\cdots&D_H(p_N,q_M)
\end{bmatrix}.
$$

---

# 3. Introduce the component-pair weights

The important point is that you **cannot simply take a weighted average of all pairwise Hellinger distances**.

Instead, introduce

$$
w_{ij}
$$

as the amount of probability mass associated with matching component \(i\) of \(P\) with component \(j\) of \(Q\).

The constraints are

$$
w_{ij}\geq0,
$$

$$
\sum_{j=1}^{M}w_{ij}=\alpha_i,
\qquad i=1,\ldots,N,
$$

and

$$
\sum_{i=1}^{N}w_{ij}=\beta_j,
\qquad j=1,\ldots,M.
$$

These are exactly the constraints in the Mercurio & Singla formulation. 

Thus \(W=[w_{ij}]\) is effectively a **transport/coupling matrix between the two sets of Gaussian components**.

---

# 4. Solve the linear program

The GMM Hellinger distance proposed in the paper is obtained from

$$
\boxed{
D_H(P,Q)
=
\min_{W}
\sum_{i=1}^{N}
\sum_{j=1}^{M}
w_{ij}D_H(p_i,q_j)
}
$$

subject to

$$
\boxed{
w_{ij}\geq0
}
$$

$$
\boxed{
\sum_{j=1}^{M}w_{ij}=\alpha_i
}
$$

$$
\boxed{
\sum_{i=1}^{N}w_{ij}=\beta_j
}
$$

The paper explicitly specializes its general mixture-distance LP to Hellinger distance in Eqs. (25)–(28). 

Because both the objective and constraints are linear, this is a convex linear-programming problem, so an obtained optimum is globally optimal. The paper also notes that \(w_{ij}=\alpha_i\beta_j\) is always a feasible starting solution. 

---

# 5. Algorithm

### **Algorithm: Hellinger Distance Between Two GMM PDFs**

**Input:**

$$
P=\{(\alpha_i,\mu_i^P,\Sigma_i^P)\}_{i=1}^{N}
$$

$$
Q=\{(\beta_j,\mu_j^Q,\Sigma_j^Q)\}_{j=1}^{M}
$$

**Output:**

$$
D_H(P,Q)
$$

---

**Step 1 — Initialize**

Set

$$
D\leftarrow \mathbf 0_{N\times M}.
$$

---

**Step 2 — Compute component distances**

For

$$
i=1,\ldots,N
$$

and

$$
j=1,\ldots,M:
$$

1. Compute

$$
\Sigma_{ij}
=
\frac12
(\Sigma_i^P+\Sigma_j^Q).
$$

2. Compute

$$
\Delta\mu_{ij}
=
\mu_i^P-\mu_j^Q.
$$

3. Compute the Gaussian Bhattacharyya coefficient

$$
B_{ij}
=
\frac{
|\Sigma_i^P|^{1/4}
|\Sigma_j^Q|^{1/4}
}{
|\Sigma_{ij}|^{1/2}
}
\exp
\left(
-\frac12
\Delta\mu_{ij}^{T}
\Sigma_{ij}^{-1}
\Delta\mu_{ij}
\right).
$$

4. Compute

$$
D_{ij}
=
\sqrt{1-B_{ij}}.
$$

---

**Step 3 — Formulate LP**

Create \(NM\) optimization variables

$$
w_{11},w_{12},\ldots,w_{NM}.
$$

Minimize

$$
J(W)
=
\sum_{i=1}^{N}
\sum_{j=1}^{M}
w_{ij}D_{ij}.
$$

Subject to

$$
\sum_{j=1}^{M}w_{ij}=\alpha_i,
\quad
i=1,\ldots,N,
$$

$$
\sum_{i=1}^{N}w_{ij}=\beta_j,
\quad
j=1,\ldots,M,
$$

$$
w_{ij}\geq0.
$$

---

**Step 4 — Solve LP**

Obtain

$$
W^\star
=
\arg\min_W J(W).
$$

---

**Step 5 — Return distance**

$$
\boxed{
D_H(P,Q)
=
\sum_{i=1}^{N}
\sum_{j=1}^{M}
w_{ij}^\star D_{ij}
}
$$

This is the GMM probabilistic distance used in the Mercurio–Singla conjunction-screening approach. 

---

## Pseudocode

```text
Algorithm Hellinger_GMM(P, Q)

Input:
    P = {alpha[i], muP[i], SigmaP[i]}, i = 1,...,N
    Q = {beta[j],  muQ[j], SigmaQ[j]}, j = 1,...,M

Output:
    DH

1.  Create D[N,M]

2.  FOR i = 1,...,N
3.      FOR j = 1,...,M

4.          Sigma_ij = 0.5 * (SigmaP[i] + SigmaQ[j])

5.          delta_mu = muP[i] - muQ[j]

6.          B_ij =
                |SigmaP[i]|^(1/4)
                |SigmaQ[j]|^(1/4)
                --------------------------------
                |Sigma_ij|^(1/2)
                *
                exp(-0.5 * delta_mu^T
                           * inv(Sigma_ij)
                           * delta_mu)

7.          D[i,j] = sqrt(1 - B_ij)

8.      END FOR
9.  END FOR

10. Solve the linear program:

        minimize
            sum_i sum_j w[i,j] * D[i,j]

        subject to
            sum_j w[i,j] = alpha[i]     for every i
            sum_i w[i,j] = beta[j]      for every j
            w[i,j] >= 0

11. Let W* be the optimal solution.

12. DH = sum_i sum_j W*[i,j] * D[i,j]

13. RETURN DH
```

### Important distinction for your conjunction-screening work

There are **two different quantities** here:

$$
\boxed{\text{Gaussian component Hellinger distance}}
$$

versus

$$
\boxed{\text{GMM Hellinger distance obtained through the LP}}.
$$

The first is an analytic expression involving two means and two covariance matrices. The second requires solving the \(N\times M\) weight-assignment problem. Mercurio & Singla specifically use this LP formulation because direct numerical integration of the Hellinger integral for non-Gaussian/GMM PDFs would be computationally expensive. 

This is also why the method fits their **kd-tree probabilistic nearest-neighbor search**: each RSO can have a GMM uncertainty representation, while the distance between two GMMs can still be evaluated through the component-wise Gaussian closed forms plus an LP rather than numerical integration. 

**One subtlety:** the lecture's Hellinger definition is the standard PDF-level metric, whereas the Mercurio–Singla GMM construction is a **mixture-distance extension using the LP coupling**. It should therefore be described in your thesis as the *LP-based Hellinger distance between GMMs*, rather than implying that the LP expression is obtained simply by analytically integrating \(\sqrt{P(x)Q(x)}\) for the two full mixture PDFs.
