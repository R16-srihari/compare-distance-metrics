## Algorithm: Riemannian Distance Between Two GMMs

### Input

Two \(K\)-component GMMs:

$$
G_1=\{(\pi_k,\mu_k,\Sigma_k)\}_{k=1}^{K}
$$

and

$$
G_2=\{(\tilde{\pi}_k,\tilde{\mu}_k,\tilde{\Sigma}_k)\}_{k=1}^{K},
$$

where

* \(\pi_k,\tilde{\pi}_k\) are mixture coefficients,
* \(\mu_k,\tilde{\mu}_k\in\mathbb R^n\) are component means,
* \(\Sigma_k,\tilde{\Sigma}_k\in\mathbb R^{n\times n}\) are positive-definite covariance matrices.

The method assumes both GMMs have \(K\) components. The embedding maps a \(K\)-component \(n\)-dimensional GMM into \(\operatorname{SPD}_{K(n+1)}\). 

---

### Step 1 — Construct the SPD embedding of \(G_1\)

For every component \(k=1,\ldots,K\), construct

$$
A_k=\Sigma_k+\pi_k\mu_k\mu_k^T,
$$

$$
X_k=\pi_k\mu_k,
$$

and

$$
B_k=\pi_k.
$$

Then construct the block matrices

$$
A=\operatorname{diag}(A_1,A_2,\ldots,A_K),
$$

$$
X=\operatorname{diag}(X_1,X_2,\ldots,X_K),
$$

$$
B=\operatorname{diag}(\pi_1,\pi_2,\ldots,\pi_K).
$$

The SPD embedding is

$$
\boxed{
S_1=
\begin{bmatrix}
A&X\\
X^T&B
\end{bmatrix}.
}
$$

This is the embedding defined by the paper's Theorem 1. 

---

### Step 2 — Construct the SPD embedding of \(G_2\)

Similarly,

$$
\tilde A_k
=
\tilde\Sigma_k+
\tilde\pi_k\tilde\mu_k\tilde\mu_k^T,
$$

$$
\tilde X_k
=
\tilde\pi_k\tilde\mu_k,
$$

$$
\tilde B_k=\tilde\pi_k.
$$

Construct

$$
\tilde A
=
\operatorname{diag}
(\tilde A_1,\ldots,\tilde A_K),
$$

$$
\tilde X
=
\operatorname{diag}
(\tilde X_1,\ldots,\tilde X_K),
$$

$$
\tilde B
=
\operatorname{diag}
(\tilde\pi_1,\ldots,\tilde\pi_K).
$$

Then

$$
\boxed{
S_2=
\begin{bmatrix}
\tilde A&\tilde X\\
\tilde X^T&\tilde B
\end{bmatrix}.
}
$$

Thus,

$$
S_1,S_2\in\operatorname{SPD}_{K(n+1)}.
$$

---

### Step 3 — Compute the AIRM distance

Calculate

$$
M=S_1^{-1/2}S_2S_1^{-1/2}.
$$

The AIRM distance is

$$
\boxed{
d_{\mathrm{R}}(G_1,G_2)
=
\left\|
\log(M)
\right\|_F
}
$$

or explicitly,

$$
\boxed{
d_{\mathrm{R}}(G_1,G_2)
=
\left\|
\log
\left(
S_1^{-1/2}
S_2
S_1^{-1/2}
\right)
\right\|_F.
}
$$

This follows directly from the closed-form geodesic distance on the SPD manifold. 

---

# Pseudocode

```text
Algorithm: Riemannian Distance Between Two GMMs

Input:
    G1 = {(πk, μk, Σk), k = 1,...,K}
    G2 = {(π̃k, μ̃k, Σ̃k), k = 1,...,K}

Output:
    dR = Riemannian distance between G1 and G2


1.  Initialize matrices A1, X1, B1

2.  FOR k = 1 TO K

3.      Ak = Σk + πk μk μkᵀ

4.      Xk = πk μk

5.      Bk = πk

6.      Store Ak, Xk and Bk as diagonal blocks

7.  END FOR

8.  Construct
        A1 = block_diag(A1, A2, ..., AK)

9.  Construct
        X1 = block_diag(X1, X2, ..., XK)

10. Construct
        B1 = diag(π1, π2, ..., πK)

11. Construct the SPD embedding
        S1 = [ A1   X1
               X1ᵀ  B1 ]


12. Initialize matrices A2, X2, B2

13. FOR k = 1 TO K

14.     Ãk = Σ̃k + π̃k μ̃k μ̃kᵀ

15.     X̃k = π̃k μ̃k

16.     B̃k = π̃k

17.     Store Ãk, X̃k and B̃k as diagonal blocks

18. END FOR

19. Construct
        A2 = block_diag(Ã1, Ã2, ..., ÃK)

20. Construct
        X2 = block_diag(X̃1, X̃2, ..., X̃K)

21. Construct
        B2 = diag(π̃1, π̃2, ..., π̃K)

22. Construct the SPD embedding
        S2 = [ A2   X2
               X2ᵀ  B2 ]


23. Compute
        M = S1^(-1/2) S2 S1^(-1/2)

24. Compute the matrix logarithm
        L = log(M)

25. Compute
        dR = ||L||F

26. RETURN dR
```

### Compact mathematical flow

The entire algorithm can be summarized as

$$
\boxed{
G_1
\xrightarrow{\ f\ }
S_1
}
\qquad
\boxed{
G_2
\xrightarrow{\ f\ }
S_2
}
$$

followed by

$$
\boxed{
d_R(G_1,G_2)
=
d_{\mathrm{AIRM}}(S_1,S_2)
=
\left\|
\log
\left(
S_1^{-1/2}S_2S_1^{-1/2}
\right)
\right\|_F
}
$$

The paper notes that this AIRM is applicable to GMMs with arbitrary means and mixing weights; for the general case it provides a lower bound to the pullback distance on the GMM manifold. 
