# Order-2 analogue of the order-3 discriminant obstruction: what exists, what is closed, what is open

Session: Fable, branch `claude/tender-archimedes-q532j1`. Computations: `eiglat.py` (exact Z-lattices with python-flint).

## 1. The order-3 mechanism, abstractly (from conway99-c3-experimental/pdf/preuve_c3.tex)

Z_3[C_3]-module F = E_3 Z_3^99, free, with a unimodular form. Two specialisations: the trivial character
(Z-lattice L = ker_Z(Q - 3I), Gram G, d = det G) and the character omega (Hermitian Z[omega]-lattice, d').
Identities N = BKB^T, GK = 77I give d = 7^a 11^b and d' = 7^c. The all-ones vector, invariant, of norm 33,
forces b odd. The two specialisations are congruent modulo the ramified prime (1 - omega), so d = d' in
F_3^*/F_3^{*2}. But 7 = 1 and 11 = -1 (mod 3): d = 2, d' = 1. Contradiction.
Three ingredients: a non-rational character, an asymmetric vector (1, giving the 11), and an arithmetic
coincidence (7 invisible mod 3, 11 visible).

## 2. The order-2 setting (computed on Paley(9) and BvLS, `eiglat.py`)

For theta in {r, s}: M = ker_Z(A - theta I) splits as L+ (invariant) and L- (anti-invariant), glued with
index 2^(rank/2); M mod 2 is a free F_2[t]-module in every case; all four lattices are even.

    BvLS theta=4:  det M = 3^168, det L+ = 2^66 3^88, det L- = 2^66 3^80, glue 2^66, free
    BvLS theta=-5: det M = 3^169, det L+ = 2^55 3^89, det L- = 2^55 3^80, glue 2^55, free
    Paley9:        det M = 3^4,   det L+ = 2^2 3^2,   det L- = 2^2 3^2,   glue 2^2,  free

For (99,14,1,2), r-side (rank 27 each): det L- = 2^27 7^{rho_S} with rho_S = rank_7(S - 3I), and
det L+ = 2^27 7^{rho_+} 11^1 (the 11 exactly once: it is rank_11 of the rank-one term 2 1 s^T of
77 E_3 restricted to the invariant part; this sharpens "odd").
Milgram on the two even lattices with the 2-parts cancelling through the glue gives, for the odd parts,

    gamma_7(L+) gamma_7(L-) gamma_11(L+) = i^27 = -i,

and since 11 = 3 (mod 4) gives gamma_11 = +-i, the only consequence is rho_+ = rho_S (mod 2). The Legendre
signs of the two 7-parts are free and absorb everything. Verified consistent on BvLS (3-adic valuations
88+80 even for rank 66, 89+80 odd for rank 55, as i^66 = -1, i^55 = -i require).

## 3. This is c2-experimental notes 09, 10, 21 (found after deriving it)

The team has the same decomposition and, with the operator-twisted Brown invariants, the stronger statement
that rank_7(S - 3I), rank_7(S_out - 3I), rank_7(Q_out - 3I) and rho_+ are all even, and (involution-free)
rank_7(A - 3I) = 1 + rho is odd with D_7(N_3) of minus type. Their verdict, which section 2 confirms
independently: genus-level arithmetic gives parities and never a contradiction.

**First point of divergence, legitimately from the parameters.**
(a) Structural: C_2 has only rational characters. Both specialisations are Z-lattices, and for Z-lattices
the product formula of the whole (N_3) implies every compatibility of the parts; the omega-side of C_3
carried an invariant modulo norms that Z-genera cannot see.
(b) Arithmetic: 7 = -1 (mod 4) is visible 2-adically (7 is not a 2-adic square), so the unknown 7-ranks
absorb the 11; in C_3, 7 = 1 (mod 3) was invisible. The parallel "11 is not a norm from Q(sqrt 2)" is
true but there is no natural Z[sqrt 2]-structure on L- to make it an invariant.
So there is no order-2 analogue of the C_3 obstruction at the genus level. This is a structural fact, not a
failed search.

## 4. Mechanism 2 (binary quadratic form on the eigen-code): closed

K = N_3 mod 2 is a 54-dimensional even-weight LCD code with the F_2-quadratic form Q = wt/2; (K, Q) is
nondegenerate with Arf = 1 (from det N_3 = 3 mod 8, using rho even). t acts as an isometry, K is free over
F_2[C_2], and K^t = (1+t)K is a Lagrangian on which Q is a linear functional. A Lagrangian with a linear
functional imposes no condition on Arf. Nothing survives.

## 5. What remains open (the only arithmetic hook), stated as a mechanism

* Structure created by the involution: D_7(N_3) = D+ perp D- (7-adically the invariant and anti-invariant
  parts are orthogonal), quadratic spaces over F_7 of even dimensions rho_+ and rho_S, whose types
  multiply to minus (parameter-forced). Individually the types are unknown.
* Compatibility condition: a quadratic space of minus type and dimension 2m has no totally singular
  subspace of dimension m; of plus type it has one. D- is the nondegenerate quotient of (F_7^49, W = 2S + I)
  and D+ of (F_7^50, 2B + I - J) restricted to invariant vectors; explicit anti-invariant vectors (shadows,
  star vectors, differences e_v - e_tv) have computable classes.
* Why (99,14,1,2) could be exceptional: rho_S <= 24 (W^2 = 0 mod 7, Witt index of F_7^49), so a family of
  13 independent mutually orthogonal singular classes in D- would force D- hyperbolic; the same in D+
  would contradict "product = minus". BvLS has gap 9, not prime: the whole 7-adic structure is specific
  to r - s = 7 prime.
* Quick falsification: on the forced part of S (line rows plus any one block type) compute the W-norms of
  all classes spanned by two or three coordinates; if no singular class exists in the forced part (the
  two-coordinate case is already empty, since -1, 5, 12 are non-squares mod 7), the mechanism needs the
  unknown far entries and is no better than the search.
* Serious route if obtained: a combinatorial construction of rho_S/2 singular orthogonal classes in D-
  from the invariant squares, plus the same for D+, with rho_S and rho_+ determined by the construction.

Verdict for this session: no new obstruction. The arithmetic side is closed for a stated structural reason;
the surviving hook (individual 7-adic types) needs precise vectors, i.e. combinatorics.
