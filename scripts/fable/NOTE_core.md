# The "core" of the involution case, and the diameter-3 objection

Session: Fable, branch `claude/tender-archimedes-q532j1`. Everything below is either proved here (short proofs
included), computed here (`core_check.py`, `runs3/`), or quoted from a team note with its location.

Setting. Gamma = srg(99,14,1,2) with an involution t whose only fixed vertex is x. N(x) = 7 edges
{e_i, -e_i}, t swaps e_i and -e_i. D = the 84 vertices at distance 2, labelled s e_i + u e_j (its two
neighbours in N(x)); t acts on D as negation. At a line c: star_c^+ = N(e_c) cap D (12 labels
(j, u) = e_c + u e_j), star_c^- = N(-e_c) cap D, far = the 60 vertices avoiding line c.

* mu_c: the matching of star_c^+ by the triangles through e_c (lambda = 1).
* beta_c: star_c^+ -> star_c^-, the unique neighbour across (mu = 2 applied to v and -e_c, e_c already
  common). tau_c := t o beta_c is a fixed-point-free involution of star_c^+ (v is never adjacent to t v
  for v in D, since a t-invariant edge would lie in a t-invariant triangle whose third vertex is fixed,
  hence x, which is not adjacent to v).
* pi: the mate pairing (j, u) <-> (j, -u).
* pivot at c: a label v with mu_c(v) = tau_c(v).

"The core" as stated by the other session: at each of the 7 lines, diameter 2 forces mu_c = tau_c = pi.

## 1. What the core identity means (proved)

**Invariant squares.** For v in D, the two common neighbours of v and -v are swapped by t (neither can be
x or in N(x)), so they form an antipodal pair {a, -a} in D. Hence every D-orbit {+-v} has a unique
*partner* orbit p({+-v}) = {+-a}, p is a fixed-point-free involution of the 42 D-orbits (a perfect matching,
21 pairs), and the 21 t-invariant induced squares are exactly v ~ a ~ -v ~ -a ~ v. (Burnside check:
2079 squares, 21 invariant, 2058 even.) An invariant square is *sib*, *one* or *dis* according as the
planes of v and a share 2, 1 or 0 lines.

**Lemma A (pivot = partner on the line).** For v = s e_i + u e_j: v is a pivot at line i iff the partner
orbit of {+-v} lies on a plane containing line i.
Proof. If the partner is a = s e_i + u' e_l (WLOG the sign of e_i is s, else exchange a and -a), then
v ~ a share the neighbour s e_i, so mu_i(v) = a, and v ~ -a = -s e_i - u' e_l means beta_i(v) = -a, so
tau_i(v) = label of t(-a) = a = mu_i(v). Conversely if mu_i(v) = tau_i(v) = (l, u') then
a := s e_i + u' e_l and -a = beta_i(v) are both neighbours of v, so {+-a} is the partner. QED

**Lemma B (sib = grid = the core identity at one label).** For v = e_c + u e_j the following are
equivalent: (a) the invariant square of v is sib; (b) mu_c(v) = tau_c(v) = pi(v); (c) the plane {c, j}
is a 3x3 grid (the 9 vertices x, +-e_c, +-e_j, +-e_c +- e_j induce K3 x K3).
Proof. (b) says e_c + u e_j ~ e_c - u e_j and e_c + u e_j ~ -e_c + u e_j; with their t-images these are
exactly the four missing edges of the grid, and exactly the square v ~ pi(v) ~ -v ~ -pi(v). QED

Consequently, with p_c the number of pivots at line c and n_sib, n_one, n_dis the numbers of D-orbits of
each type (all even, since p pairs orbits of the same type):

    sum_c p_c = 2 n_sib + n_one,     n_sib = 2 |Sigma|,

where Sigma is the graph on the 7 lines whose edges are the grid planes. The team has certified
|Sigma| <= 4 and deg Sigma <= 2 (c2-experimental notes/19 section 9, star lemma + DRAT), so in any
hypothetical Gamma:

* at every line, at most 4 of the 12 labels satisfy the core identity mu_c(v) = tau_c(v) = pi(v);
* at least 34 of the 42 D-orbits have a non-sib invariant square;
* n_one <= 42 - n_sib, so the pivot total is sum_c p_c <= 16 + n_one.

So "the core" (mu_c = tau_c = pi at every line, or even at one line) contradicts a certified bound: proving
it from "Gamma exists" is proving the theorem, and it cannot be proved one line at a time (section 2).
This is the same statement as the team's free-star lemma, global form (c2-experimental notes/20 section 0:
"every block belongs to a sister pair", noted there as equivalent to non-existence).

## 2. The core identity is independent of the block-local axioms (computed)

`core_check.py` evaluates mu_c, tau_c, pi at all 11 lines of BvLS = srg(243,22,1,2) with t = -1:

    all 11 lines: 20/20 pivots, mu_c = tau_c = pi on all 20 labels; all 220 D-vertices are sib.

So the core identity holds in the one known member of the family, which satisfies every local hypothesis
(lambda = 1, mu = 2, an involution with one fixed point, diameter 2). The block-local system at a line
(the team's (i)-(iii) of notes/20 section 3, my K2/K3/Phi) is therefore *consistent with* the core
identity. C1's twisted witnesses (notes/20, notes/103, notes/105 of conway99-c1) are diameter-3
structures satisfying the same local system with mu_c != pi: the local system is also *consistent with
the negation*. Hence:

> No argument that uses only the local axioms at a line (or at several lines, since BvLS satisfies them
> at all lines simultaneously) can prove the core identity. Any proof must use m = 7 globally, exactly
> as the certified bound |Sigma| <= 4 does (it is false for m = 11: BvLS has deg Sigma = 10).

## 3. The diameter-3 objection, answered

The concern: the team used diameter-3 graphs to kill hypotheses about a diameter-2 graph, so good leads
may have been killed by accident.

* What a diameter-3 witness proves: that the *local form* of a lead (the lead as a statement about the
  block-local relaxation) does not follow from that relaxation. C1's notes/20 records the verdicts as
  "local form killed, SRG form open", which is the correct bookkeeping. Nothing about the SRG was
  refuted by those witnesses.
* What the SRG forms are worth: by section 1, the SRG form of the core (and of every lead of the type
  "the local matchings are forced to be pi / to coincide") is equivalent to the theorem itself. They are
  not leads that were lost; they are restatements of the goal. Section 2 shows they cannot be reached
  from the local relaxation at all, whatever witness is used.
* The one hypothesis the team's per-type local instances genuinely relax is mu = 2 between two far
  vertices of the same block (both stages). The exact whole-graph model with this included
  (`sat_full.py`, the team's `sat_whole.py` with the block-0 type fixed, ~578k vars / 1.31M clauses) was
  run here on a free star (0 pivots), a 4-pivot, an 8-pivot and a full-block (12-pivot) type, plus the
  free star with the all-dis clauses: all UNKNOWN after >= 32 CPU-minutes each (`runs3/`). The team's
  quotient stage (Stage Q, kissat and CP-SAT, 2 h, cases sib/one/dis) and its U0-row cubes are also
  UNKNOWN (conway99-involution/results/HANDOFF_2026-09-22.md). So restoring the diameter-2 hypothesis in
  full does not make the local statements decidable in-session; it does not refute them either.

Verdict: no lead was killed wrongly. The diameter-3 witnesses only ever killed proof strategies that
used the relaxation, and section 2 shows every such strategy for the core is dead in principle.

## 4. What a real attack on the core needs

The structural difference between m = 7 and m = 11 is not visible at one line. It is global:
BvLS has all 220 invariant squares sib; a hypothetical Gamma has at least 34 of 42 D-orbits non-sib, and
no configuration with a non-sib invariant square in an srg(v,k,1,2) is known. Two cheap global counts,
both automatic consequences of the spectrum (checked against tr C^3 and tr S^3, C = symmetric quotient on
D-orbits with spectrum {12, (-2)^6, 3^20, (-4)^15}, S_DD = its signing with spectrum {0^7, 3^20, (-4)^15}):

    #quotient triangles through a partner pair   = 42 - (sum_c p_c)/2
    #A'-triangles (single edges only)             = 126 + sum_c p_c,
        balanced (lift to two triangles of Gamma)  = 28 + (sum_c p_c)/2,
        unbalanced (lift to a twisted hexagon)     = 98 + (sum_c p_c)/2.

These are consistent for every value of sum_c p_c (0 to 16 + n_one) and give no contradiction on their
own; they are recorded so that they are not re-derived.

The smallest open statement that would close the case is the team's free-star lemma, local form:
RSF(star(c)) is UNSAT when no grid contains line c (c2-experimental notes/20). Its negation and its
truth are both consistent with everything known locally (section 2), so it is a global finite
computation, not a lemma to be found by hand. The decisive finite computation currently running in the
team is Route (C) (embedding of the rank-22 lattice 3I - S into the 156 odd unimodular lattices of rank
24 without norm-1 vectors); nothing in this note changes its status.
