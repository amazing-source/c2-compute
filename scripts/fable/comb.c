/*
 * step12.c — block-local system of the involution case of srg(2m^2+1, 2m, 1, 2), one block type.
 *
 * Step 1: enumerate iota (fixed-point-free involutions on the non-mu pairs) with cross counts K3.
 * Step 2: for each iota, assign the iota-pairs (with orientation) to the far orbits under the exact K2 counts,
 *         the local bounds, the pairwise (far,far) bounds, the exact row weights, and the real residual
 *         completion of the far-far sign block T (residual dimension = resdim).
 *
 * Data file: m, then mu[nl], tau[nl], then Gpinv (nk x nk doubles) with nk = m + nl, nl = 2(m-1).
 * Optional replay file: phi[root] as pair index (nr ints), then T[o][o'] (no x no ints) — restricts the search.
 * Usage: step12 datafile [max_iota] [time_limit_s] [verbose] [node_limit2] [replayfile]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAXNL 20
#define MAXNP 180
#define MAXNR 180
#define MAXNO 90
#define MAXNK 31
#define MAXRES 25

static int m_, nl, np, nr, no, nk, resdim, rowweight;   /* rowweight = 2m-4 nonzero D-entries per far row */
static double RVAL;                                     /* r = norm^2 of a D-orbit vector */
static int mu[MAXNL], tau[MAXNL], pi_[MAXNL], hh[MAXNL], pivot[MAXNL], line_[MAXNL];
static int pair_v[MAXNP][2], pair_index[MAXNL][MAXNL];
static int root_lab[MAXNR][2], root_index[MAXNL][MAXNL], troot[MAXNR];
static int orb_rep[MAXNO], orb_of_root[MAXNR];
static int K2[MAXNL][MAXNL], K3[MAXNL][MAXNL];
static double Gpinv[MAXNK][MAXNK];
static double TOL = 1e-10;
static int verbose = 0;
static double time_limit = 0;
static clock_t t_start;
static int replay = 0;
static int phi_true[MAXNR];
static int T_true[MAXNO][MAXNO];

static int in_pair(int p, int v) { return pair_v[p][0] == v || pair_v[p][1] == v; }
static int in_root(int r, int v) { return root_lab[r][0] == v || root_lab[r][1] == v; }

/* ------------------------------------------------------------------ step 1: iota */
static int match_[MAXNP];
static int K3rem[MAXNL][MAXNL];
static int (*contrib)[MAXNP][8][2];
static int pairs_with[MAXNL][MAXNL];
static long long nodes1 = 0;
static int max_iota = 1000000;
static int (*iotas)[MAXNP];
static int n_iota = 0;
static int stop1 = 0;

static void apply1(int p, int q, int sign) {
    for (int i = 0; i < 8; i++) K3rem[contrib[p][q][i][0]][contrib[p][q][i][1]] -= sign;
    if (sign == 1) { match_[p] = q; match_[q] = p; } else { match_[p] = -1; match_[q] = -1; }
}

static int coverable1(void) {
    int unmatched_cnt[MAXNL], single[MAXNL];
    for (int v = 0; v < nl; v++) {
        int c = 0, s = -1;
        for (int i = 0; i < nl - 2; i++) { int r = pairs_with[v][i]; if (match_[r] < 0) { c++; s = r; } }
        unmatched_cnt[v] = c; single[v] = (c == 1) ? s : -1;
    }
    for (int v = 0; v < nl; v++) for (int w = 0; w < nl; w++) {
        int b = K3rem[v][w];
        if (b <= 0) continue;
        int t = pi_[w];
        int ut = unmatched_cnt[t];
        if (ut == 0) return 0;
        int cnt = unmatched_cnt[v];
        if (ut == 1) { int s0 = single[t]; if (in_pair(s0, v)) cnt--; }
        if (cnt < b) return 0;
    }
    return 1;
}

static void rec1(void) {
    if (stop1) return;
    nodes1++;
    if (time_limit > 0 && (nodes1 & 4095) == 0) {
        if ((double)(clock() - t_start) / CLOCKS_PER_SEC > time_limit) { stop1 = 1; return; }
    }
    int best_p = -1, best_n = 1 << 30, best_list[MAXNP];
    int any = 0;
    for (int p = 0; p < np; p++) {
        if (match_[p] >= 0) continue;
        any = 1;
        int lst[MAXNP], n = 0;
        for (int q = 0; q < np; q++) {
            if (q == p || match_[q] >= 0) continue;
            int a = p < q ? p : q, b = p < q ? q : p;
            int ok = 1;
            for (int i = 0; i < 8; i++) if (K3rem[contrib[a][b][i][0]][contrib[a][b][i][1]] < 1) { ok = 0; break; }
            if (ok) lst[n++] = q;
        }
        if (n < best_n) { best_n = n; best_p = p; memcpy(best_list, lst, n * sizeof(int)); if (n == 0) return; }
    }
    if (!any) {
        if (n_iota < max_iota) memcpy(iotas[n_iota], match_, sizeof(int) * np);
        n_iota++;
        if (verbose) { printf("  iota #%d at node %lld (%.1fs)\n", n_iota, nodes1, (double)(clock()-t_start)/CLOCKS_PER_SEC); fflush(stdout); }
        if (n_iota >= max_iota) stop1 = 2;
        return;
    }
    int p = best_p;
    for (int i = 0; i < best_n; i++) {
        int q = best_list[i];
        int a = p < q ? p : q, b = p < q ? q : p;
        apply1(a, b, 1);
        if (coverable1()) rec1();
        apply1(a, b, -1);
        if (stop1) return;
    }
}

/* ------------------------------------------------------------------ step 2 */
static int ipair[MAXNO][2];
static int ipair_used[MAXNO];
static int pair_used_[MAXNP];
static int opts_all[MAXNO][MAXNP*MAXNP][2];
static int phi[MAXNR];
static int K2rem[MAXNL][MAXNL];
static int K3rem2[MAXNL][MAXNL];
static int (*cand)[MAXNP*MAXNP][2]; static int ncand[MAXNO];
static int rootsleft[MAXNL];
static int assigned[MAXNO];
static int alist[MAXNO], nassigned;
static int gvec_[MAXNO][MAXNK];
static double kk[MAXNO][MAXNO];
static double rho_[MAXNO];
static double rc[MAXNO][MAXRES];
static int ncols;
static int basis_pos[MAXRES];
static int Tm[MAXNO][MAXNO];
static int tnz[MAXNO];
static int sigw[MAXNO];
static long long nodes2 = 0, sol2 = 0;
static int stop2 = 0;
static long long node_limit2 = 0;
static int leaf_print = 1;
static int maxdepth2 = 0;
static double sum_bn[MAXNO+1]; static long long cnt_bn[MAXNO+1]; static int nores = 0; static int usepsd = 0;
static double Om[MAXNK][MAXNK]; static double Omrem[MAXNK][MAXNK]; static long long psd_prunes = 0;

static int local_ok_root(int z, int P, int tP) {
    int a = root_lab[z][0], b = root_lab[z][1];
    for (int lab = 0; lab < nl; lab++) {
        int plab = pi_[lab];
        if (lab == a || lab == b) {
            if (in_pair(P, lab) + in_pair(tP, plab) > 1) return 0;
        } else {
            if ((plab == a || plab == b) + in_pair(P, lab) + in_pair(tP, plab) > 2) return 0;
        }
    }
    for (int v = 0; v < nl; v++) {
        int target = in_pair(P, v) ? 1 : 2;
        if ((v == a || v == b) + in_pair(P, mu[v]) + in_pair(tP, tau[v]) > target) return 0;
    }
    return 1;
}

static int local_ok_orbit(int z, int P, int tP) {
    if (P == tP) return 0;
    int inter = 0, cv = -1;
    for (int i = 0; i < 2; i++) if (in_pair(tP, pair_v[P][i])) { inter++; cv = pair_v[P][i]; }
    if (inter > 1) return 0;
    if (inter == 1 && pivot[cv]) return 0;
    return local_ok_root(z, P, tP) && local_ok_root(troot[z], tP, P);
}

static int common_known(int z1, int z2, int P1, int tP1, int P2, int tP2) {
    int c = 0;
    for (int i = 0; i < 2; i++) if (in_root(z2, root_lab[z1][i])) c++;
    for (int i = 0; i < 2; i++) if (in_pair(P2, pair_v[P1][i])) c++;
    for (int i = 0; i < 2; i++) if (in_pair(tP2, pair_v[tP1][i])) c++;
    return c;
}

/* allowed T (bitmask: bit (t+1)) between the new orbit (z,P,tP) and assigned position j */
static int allowed_T(int o, int z, int P, int tP, int j) {
    int o2 = alist[j];
    int z2 = orb_rep[o2];
    int P2 = phi[z2], tP2 = phi[troot[z2]];
    int c1 = common_known(z, z2, P, tP, P2, tP2);
    int c2 = common_known(z, troot[z2], P, tP, tP2, P2);
    if (c1 > 2 || c2 > 2) return 0;
    int mask = 0;
    for (int x = 0; x <= (c1 < 2 ? 1 : 0); x++)
        for (int y = 0; y <= (c2 < 2 ? 1 : 0); y++) mask |= 1 << ((x - y) + 1);
    if (replay) mask &= 1 << (T_true[o][o2] + 1);
    return mask;
}

static void apply2_counts(int z, int P, int tP, int sign) {
    int tz = troot[z];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
        K3rem2[pair_v[P][i]][pi_[pair_v[tP][j]]] -= sign;
        K3rem2[pair_v[tP][j]][pi_[pair_v[P][i]]] -= sign;
    }
    for (int k = 0; k < 2; k++) {
        int lab = root_lab[z][k];
        rootsleft[lab] -= sign;
        for (int i = 0; i < 2; i++) K2rem[lab][pair_v[P][i]] -= sign;
        int tlab = root_lab[tz][k];
        rootsleft[tlab] -= sign;
        for (int i = 0; i < 2; i++) K2rem[tlab][pair_v[tP][i]] -= sign;
    }
    if (sign == 1) { phi[z] = P; phi[tz] = tP; } else { phi[z] = -1; phi[tz] = -1; }
}

static int k2_ok(int z, int P, int tP) {
    int tz = troot[z];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
        if (K3rem2[pair_v[P][i]][pi_[pair_v[tP][j]]] < 1) return 0;
        if (K3rem2[pair_v[tP][j]][pi_[pair_v[P][i]]] < 1) return 0;
    }
    for (int k = 0; k < 2; k++) {
        int lab = root_lab[z][k];
        for (int i = 0; i < 2; i++) if (K2rem[lab][pair_v[P][i]] < 1) return 0;
        int tlab = root_lab[tz][k];
        for (int i = 0; i < 2; i++) if (K2rem[tlab][pair_v[tP][i]] < 1) return 0;
    }
    return 1;
}

static void make_gvec(int z, int P, int tP, int *g) {
    for (int i = 0; i < nk; i++) g[i] = 0;
    for (int k = 0; k < 2; k++) {
        int lab = root_lab[z][k];
        g[line_[lab]] = -((lab % 2 == 0) ? 1 : -1);
    }
    for (int i = 0; i < 2; i++) g[m_ + pair_v[P][i]] -= 1;
    for (int i = 0; i < 2; i++) g[m_ + pair_v[tP][i]] += 1;
}

static double gdot(const int *g1, const int *g2) {
    double s = 0;
    for (int i = 0; i < nk; i++) {
        if (!g1[i]) continue;
        double row = 0;
        for (int j = 0; j < nk; j++) if (g2[j]) row += Gpinv[i][j] * g2[j];
        s += g1[i] * row;
    }
    return s;
}

static int solve_small(int n, double A[MAXRES][MAXRES], double *b, double *x) {
    double M[MAXRES][MAXRES + 1];
    for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) M[i][j] = A[i][j]; M[i][n] = b[i]; }
    for (int c = 0; c < n; c++) {
        int piv = c; double best = fabs(M[c][c]);
        for (int r = c + 1; r < n; r++) if (fabs(M[r][c]) > best) { best = fabs(M[r][c]); piv = r; }
        if (best < 1e-12) return 0;
        if (piv != c) for (int j = 0; j <= n; j++) { double t = M[c][j]; M[c][j] = M[piv][j]; M[piv][j] = t; }
        for (int r = 0; r < n; r++) if (r != c) {
            double f = M[r][c] / M[c][c];
            for (int j = c; j <= n; j++) M[r][j] -= f * M[c][j];
        }
    }
    for (int i = 0; i < n; i++) x[i] = M[i][n] / M[i][i];
    return 1;
}


static double min_eig_sym(int n, double M[MAXNK][MAXNK]) {
    double A[MAXNK][MAXNK];
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) A[i][j] = M[i][j];
    for (int sweep = 0; sweep < 60; sweep++) {
        double off = 0;
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
        if (off < 1e-18) break;
        for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) {
            if (fabs(A[p][q]) < 1e-14) continue;
            double theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
            double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1));
            double cs = 1 / sqrt(t * t + 1), sn = t * cs;
            for (int k = 0; k < n; k++) { double akp = A[k][p], akq = A[k][q]; A[k][p] = cs * akp - sn * akq; A[k][q] = sn * akp + cs * akq; }
            for (int k = 0; k < n; k++) { double apk = A[p][k], aqk = A[q][k]; A[p][k] = cs * apk - sn * aqk; A[q][k] = sn * apk + cs * aqk; }
        }
    }
    double mn = A[0][0];
    for (int i = 1; i < n; i++) if (A[i][i] < mn) mn = A[i][i];
    return mn;
}

static void rec2(void);

static void try_orbit(int o, int P, int tP) {
    int z = orb_rep[o];
    int pos = nassigned;
    int amask[MAXNO];
    for (int j = 0; j < nassigned; j++) { amask[j] = allowed_T(o, z, P, tP, j); if (!amask[j]) return; }
    int g[MAXNK];
    make_gvec(z, P, tP, g);
    double kkrow[MAXNO];
    for (int j = 0; j < nassigned; j++) kkrow[j] = gdot(g, gvec_[j]);
    double self = gdot(g, g);
    double rho = RVAL - self;
    if (rho < -TOL) return;
    if (rho < 0) rho = 0;
    int sw = 0; for (int i = 0; i < nl; i++) if (g[m_ + i]) sw++;
    int need_nz = rowweight - sw;
    int nb = nores ? 0 : ncols;
    double A[MAXRES][MAXRES];
    for (int i = 0; i < nb; i++) for (int j = 0; j < nb; j++) A[i][j] = rc[basis_pos[i]][j];
    /* enumerate T for the basis orbits: 3^nb combinations (nb <= 4 in the m = 7 case; replay restricts) */
    /* odometer over the allowed T values of the basis orbits */
    int av[MAXRES][3], nav[MAXRES], oi[MAXRES];
    long long ncomb = 1;
    for (int t = 0; t < nb; t++) { nav[t] = 0; for (int tv = -1; tv <= 1; tv++) if (amask[basis_pos[t]] & (1 << (tv + 1))) av[t][nav[t]++] = tv; ncomb *= nav[t]; oi[t] = 0; }
    if (ncomb > 10000000) { fprintf(stderr, "too many basis combinations (nb=%d)\n", nb); exit(5); }
    for (long long idx = 0; idx < ncomb; idx++) {
        int cnt[MAXRES];
        { long long x = idx; for (int t = 0; t < nb; t++) { cnt[t] = av[t][x % nav[t]]; x /= nav[t]; } }
        double b[MAXRES], c[MAXRES];
        for (int t = 0; t < nb; t++) b[t] = -cnt[t] - kkrow[basis_pos[t]];
        if (nb > 0 && !solve_small(nb, A, b, c)) continue;
        double proj2 = 0; for (int t = 0; t < nb; t++) proj2 += c[t] * c[t];
        double extra = rho - proj2;
        if (extra < -TOL) continue;
        int newdim = (extra > TOL);
        if (nores) { newdim = 0; }
        if (newdim && nb >= resdim) continue;
        int ok = 1, nz = 0;
        int Trow[MAXNO];
        for (int j = 0; j < nassigned && ok; j++) if (nores) Trow[j] = 0; else {
            int isb = -1; for (int t = 0; t < nb; t++) if (basis_pos[t] == j) isb = t;
            if (isb >= 0) Trow[j] = cnt[isb];
            else {
                double ip = 0; for (int t = 0; t < nb; t++) ip += c[t] * rc[j][t];
                double tv = -ip - kkrow[j];
                int tr = (int)floor(tv + 0.5);
                if (fabs(tv - tr) > TOL || tr < -1 || tr > 1 || !(amask[j] & (1 << (tr + 1)))) ok = 0;
                else Trow[j] = tr;
            }
            if (ok && Trow[j]) { nz++; if (nz > need_nz) ok = 0; if (tnz[j] + 1 > rowweight - sigw[j]) ok = 0; }
        }
        if (!nores && ok && need_nz - nz > no - nassigned - 1) ok = 0;
        if (!ok) continue;
        /* commit */
        alist[pos] = o; assigned[o] = pos; nassigned++;
        if (nassigned > maxdepth2) maxdepth2 = nassigned;
        memcpy(gvec_[pos], g, sizeof(int) * nk);
        for (int j = 0; j < pos; j++) kk[pos][j] = kk[j][pos] = kkrow[j];
        kk[pos][pos] = self;
        rho_[pos] = rho; sigw[pos] = sw;
        for (int t = 0; t < MAXRES; t++) rc[pos][t] = 0;
        for (int t = 0; t < nb; t++) rc[pos][t] = c[t];
        int oldncols = ncols;
        if (newdim) { rc[pos][nb] = sqrt(extra); basis_pos[nb] = pos; ncols = nb + 1; }
        for (int j = 0; j < pos; j++) { Tm[pos][j] = Tm[j][pos] = Trow[j]; if (Trow[j]) tnz[j]++; }
        tnz[pos] = nz;
        apply2_counts(z, P, tP, 1);
        int psd_ok = 1;
        if (usepsd) {
            for (int i = 0; i < nk; i++) if (g[i]) for (int j = 0; j < nk; j++) if (g[j]) Omrem[i][j] -= g[i] * g[j];
            if (min_eig_sym(nk, Omrem) < -1e-6) { psd_ok = 0; psd_prunes++; }
        }
        if (psd_ok) rec2();
        if (usepsd) for (int i = 0; i < nk; i++) if (g[i]) for (int j = 0; j < nk; j++) if (g[j]) Omrem[i][j] += g[i] * g[j];
        apply2_counts(z, P, tP, -1);
        for (int j = 0; j < pos; j++) if (Trow[j]) tnz[j]--;
        ncols = oldncols;
        nassigned--; assigned[o] = -1;
        if (stop2) return;
    }
}

static void rec2(void) {
    if (stop2) return;
    nodes2++;
    if (node_limit2 && nodes2 > node_limit2) { stop2 = 1; return; }
    if (time_limit > 0 && (nodes2 & 1023) == 0 && (double)(clock() - t_start) / CLOCKS_PER_SEC > time_limit) { stop2 = 1; return; }
    if (nassigned == no) {
        for (int j = 0; j < no; j++) if (tnz[j] != rowweight - sigw[j]) return;
        sol2++;
        if (leaf_print) {
            printf("SOLUTION (sign stage) found: nodes2=%lld\n  phi:", nodes2);
            for (int r = 0; r < nr; r++) printf(" %d:(%d,%d)", r, pair_v[phi[r]][0], pair_v[phi[r]][1]);
            printf("\n  T:");
            for (int i = 0; i < no; i++) for (int j = 0; j < i; j++) if (Tm[i][j]) printf(" (%d,%d):%d", alist[i], alist[j], Tm[i][j]);
            printf("\n"); fflush(stdout);
        }
        return;
    }
    int best_o = -1, best_n = 1 << 30;
    for (int o = 0; o < no; o++) {
        if (assigned[o] >= 0) continue;
        int z = orb_rep[o];
        int n = 0;
        for (int ci = 0; ci < ncand[o]; ci++) {
            int P = cand[o][ci][0], tP = cand[o][ci][1];
            if (pair_used_[P] || pair_used_[tP]) continue;
            if (replay && (phi_true[z] != P || phi_true[troot[z]] != tP)) continue;
            if (!k2_ok(z, P, tP)) continue;
            int ok = 1;
            for (int j = 0; j < nassigned; j++) if (!allowed_T(o, z, P, tP, j)) { ok = 0; break; }
            if (!ok) continue;
            opts_all[o][n][0] = P; opts_all[o][n][1] = tP; n++;
            if (n >= best_n) break;   /* cannot beat the current best orbit */
        }
        if (verbose > 1 && nodes2 <= 2) printf("   node %lld orbit %d (root %d,%d): %d options\n", nodes2, o, root_lab[z][0], root_lab[z][1], n);
        if (n < best_n) { best_n = n; best_o = o; if (n == 0) return; }
    }
    sum_bn[nassigned] += best_n; cnt_bn[nassigned]++;
    int o = best_o;
    for (int i = 0; i < best_n; i++) {
        int P = opts_all[o][i][0], tP = opts_all[o][i][1];
        pair_used_[P] = 1; pair_used_[tP] = 1;
        try_orbit(o, P, tP);
        pair_used_[P] = 0; pair_used_[tP] = 0;
        if (stop2) return;
    }
}

static void setup(void) {
    nl = 2 * (m_ - 1); nk = m_ + nl; rowweight = 2 * m_ - 4;
    {
        int d = (int)floor(sqrt(8.0 * m_ - 7) + 0.5);
        int r = (-1 + d) / 2;
        RVAL = r;
        long long v = 2LL * m_ * m_ + 1, k = 2 * m_;
        double f = ((v - 1) - (2.0 * k + (v - 1) * (-1)) / d) / 2.0;   /* multiplicity of r */
        double g = (v - 1) - f;
        resdim = (int)floor(g / 2 + 0.5) - (3 * m_ - 3);
    }
    for (int v = 0; v < nl; v++) { pi_[v] = v ^ 1; line_[v] = v / 2 + 1; }
    for (int v = 0; v < nl; v++) { hh[v] = pi_[tau[v]]; pivot[v] = (mu[v] == tau[v]); }
    np = 0;
    for (int v = 0; v < nl; v++) for (int w = 0; w < nl; w++) pair_index[v][w] = -1;
    for (int v = 0; v < nl; v++) for (int w = v + 1; w < nl; w++) if (mu[v] != w) { pair_v[np][0] = v; pair_v[np][1] = w; pair_index[v][w] = pair_index[w][v] = np; np++; }
    for (int v = 0; v < nl; v++) { int c = 0; for (int p = 0; p < np; p++) if (in_pair(p, v)) pairs_with[v][c++] = p; if (c != nl - 2) { fprintf(stderr, "pw\n"); exit(1); } }
    nr = 0;
    for (int a = 0; a < nl; a++) for (int b = 0; b < nl; b++) root_index[a][b] = -1;
    for (int a = 0; a < nl; a++) for (int b = a + 1; b < nl; b++) if (line_[a] != line_[b]) { root_lab[nr][0] = a; root_lab[nr][1] = b; root_index[a][b] = root_index[b][a] = nr; nr++; }
    if (nr != np) { fprintf(stderr, "nr=%d np=%d\n", nr, np); exit(1); }
    for (int r = 0; r < nr; r++) troot[r] = root_index[pi_[root_lab[r][0]]][pi_[root_lab[r][1]]];
    no = 0;
    for (int r = 0; r < nr; r++) orb_of_root[r] = -1;
    for (int r = 0; r < nr; r++) if (root_lab[r][0] % 2 == 0) { orb_rep[no] = r; orb_of_root[r] = no; orb_of_root[troot[r]] = no; no++; }
    if (no != nr / 2) { fprintf(stderr, "no=%d\n", no); exit(1); }
    for (int a = 0; a < nl; a++) for (int v = 0; v < nl; v++)
        K2[a][v] = 2 - (v == a) - (v == pi_[a]) - (v == mu[a]) - (hh[v] == a);
    int mup[MAXNL]; for (int w = 0; w < nl; w++) mup[w] = pi_[mu[pi_[w]]];
    for (int v = 0; v < nl; v++) for (int w = 0; w < nl; w++)
        K3[v][w] = (hh[v] == w ? 1 : 2) - (v == w) - (hh[mu[v]] == w) - (hh[v] == mup[w]);
    contrib = malloc(sizeof(int) * MAXNP * MAXNP * 16);
    for (int p = 0; p < np; p++) for (int q = 0; q < np; q++) {
        int v1 = pair_v[p][0], v2 = pair_v[p][1], w1 = pair_v[q][0], w2 = pair_v[q][1];
        int c[8][2] = {{v1, pi_[w1]}, {v1, pi_[w2]}, {v2, pi_[w1]}, {v2, pi_[w2]}, {w1, pi_[v1]}, {w1, pi_[v2]}, {w2, pi_[v1]}, {w2, pi_[v2]}};
        memcpy(contrib[p][q], c, sizeof(c));
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s datafile [max_iota] [time_limit] [verbose] [node_limit2] [replayfile]\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("open"); return 1; }
    if (fscanf(f, "%d", &m_) != 1) return 2;
    nl = 2 * (m_ - 1); nk = m_ + nl;
    for (int i = 0; i < nl; i++) if (fscanf(f, "%d", &mu[i]) != 1) return 2;
    for (int i = 0; i < nl; i++) if (fscanf(f, "%d", &tau[i]) != 1) return 2;
    for (int i = 0; i < nk; i++) for (int j = 0; j < nk; j++) if (fscanf(f, "%lf", &Gpinv[i][j]) != 1) return 3;
    fclose(f);
    if (argc > 2) max_iota = atoi(argv[2]);
    if (argc > 3) time_limit = atof(argv[3]);
    if (argc > 4) verbose = atoi(argv[4]);
    if (argc > 5) node_limit2 = atoll(argv[5]);
    setup();
    if (getenv("NORES")) nores = 1;
    if (getenv("PSD")) usepsd = 1;
    {
        static double Gk[MAXNK][MAXNK];
        for (int i = 0; i < nk; i++) for (int j = 0; j < nk; j++) Gk[i][j] = 0;
        for (int l = 0; l < m_; l++) Gk[l][l] = RVAL + 1;
        for (int v = 0; v < nl; v++) {
            int P = m_ + v;
            Gk[0][P] = Gk[P][0] = -1;
            int l = line_[v];
            double s = (v % 2 == 0) ? 1 : -1;
            Gk[l][P] = Gk[P][l] = -s;
            Gk[P][P] = RVAL;
            for (int w = 0; w < nl; w++) if (w != v) Gk[P][m_ + w] = -((mu[v] == w ? 1 : 0) - (hh[v] == pi_[w] ? 1 : 0));
        }
        for (int i = 0; i < nk; i++) for (int j = 0; j < nk; j++) { double s = 0; for (int t = 0; t < nk; t++) s += Gk[i][t] * Gk[t][j]; Om[i][j] = (2 * RVAL + 1) * Gk[i][j] - s; Omrem[i][j] = Om[i][j]; }
        printf("Omega: min eig %.4f, trace %.1f\n", min_eig_sym(nk, Om), (double)0 + ({double tr=0; for(int i=0;i<nk;i++) tr+=Om[i][i]; tr;}));
    }
    if (argc > 6) {
        FILE *fr = fopen(argv[6], "r");
        if (!fr) { perror("replay"); return 1; }
        for (int r = 0; r < nr; r++) if (fscanf(fr, "%d", &phi_true[r]) != 1) return 6;
        for (int i = 0; i < no; i++) for (int j = 0; j < no; j++) if (fscanf(fr, "%d", &T_true[i][j]) != 1) return 6;
        fclose(fr);
        replay = 1;
    }
    printf("m=%d nl=%d np=%d nr=%d no=%d nk=%d resdim=%d rowweight=%d r=%g\n", m_, nl, np, nr, no, nk, resdim, rowweight, RVAL);
    int neg = 0;
    for (int a = 0; a < nl; a++) for (int v = 0; v < nl; v++) if (K2[a][v] < 0 || K3[a][v] < 0) neg = 1;
    if (neg) { printf("TYPE INVALID (negative K2/K3): no solution\nRESULT invalid 0 iota, 0 solutions\n"); return 0; }
    iotas = malloc(sizeof(int) * MAXNP * (size_t)(max_iota > 0 ? max_iota : 1));
    t_start = clock();
    /* precompute per-orbit candidate (P,tP) lists passing the local bounds */
    cand = malloc(sizeof(int) * 2 * (size_t)MAXNP * MAXNP * MAXNO);
    long long totc = 0;
    for (int o = 0; o < no; o++) {
        int z = orb_rep[o]; ncand[o] = 0;
        for (int P = 0; P < np; P++) for (int tP = 0; tP < np; tP++) if (P != tP && local_ok_orbit(z, P, tP)) { cand[o][ncand[o]][0] = P; cand[o][ncand[o]][1] = tP; ncand[o]++; }
        totc += ncand[o];
    }
    printf("combined search: candidates per orbit avg %.0f\n", (double)totc / no);
    for (int r = 0; r < nr; r++) phi[r] = -1;
    memcpy(K2rem, K2, sizeof(K2));
    memcpy(K3rem2, K3, sizeof(K3));
    for (int v = 0; v < nl; v++) rootsleft[v] = nl - 2;
    for (int o = 0; o < no; o++) assigned[o] = -1;
    memset(pair_used_, 0, sizeof(pair_used_));
    nassigned = 0; ncols = 0; nodes2 = 0; sol2 = 0; stop2 = 0; maxdepth2 = 0;
    memset(tnz, 0, sizeof(tnz));
    rec2();
    printf("RESULT %s: nodes %lld, maxdepth %d, solutions %lld, %.1fs\n", stop2 ? "incomplete" : "complete", nodes2, maxdepth2, sol2, (double)(clock() - t_start) / CLOCKS_PER_SEC);
    printf("psd prunes %lld\n", psd_prunes); printf("avg options per depth:"); for (int d = 0; d <= no; d++) if (cnt_bn[d]) printf(" d%d:%.1f(n=%lld)", d, sum_bn[d]/cnt_bn[d], cnt_bn[d]); printf("\n");
    return 0;
}
