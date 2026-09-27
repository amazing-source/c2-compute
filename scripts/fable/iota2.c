/*
 * iota2.c — exact-cover formulation of step 1 (iota enumeration), Knuth-style:
 * at each node choose the K3 cell (v,w) with remaining budget b in {1,2} that has the fewest branches, and
 * cover it COMPLETELY at that node: branch over single feasible coverings (b = 1) or over unordered pairs of
 * jointly feasible coverings (b = 2).  Each iota is generated exactly once.
 * A covering of (v,w) is a matching {p, q} with v in p, pi(w) in q (p != q, both unmatched).
 * Data file as for step12 (m, mu, tau, ...).  Usage: iota2 datafile [time_limit] [max_iota] [verbose] [dumpfile]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAXNL 20
#define MAXNP 180
static int m_, nl, np;
static int mu[MAXNL], tau[MAXNL], pi_[MAXNL], hh[MAXNL], line_[MAXNL];
static int pair_v[MAXNP][2];
static int K3[MAXNL][MAXNL], K3rem[MAXNL][MAXNL];
static int pairs_with[MAXNL][MAXNL];
static int match_[MAXNP];
static int (*contrib)[MAXNP][8][2];
static long long nodes = 0, n_iota = 0, max_iota = 0;
static double time_limit = 0; static clock_t t0; static int stop_ = 0, verbose = 0;
static FILE *dump = NULL;
static int in_pair(int p, int v) { return pair_v[p][0] == v || pair_v[p][1] == v; }

static void apply(int p, int q, int sign) {
    int a = p < q ? p : q, b = p < q ? q : p;
    for (int i = 0; i < 8; i++) K3rem[contrib[a][b][i][0]][contrib[a][b][i][1]] -= sign;
    if (sign == 1) { match_[p] = q; match_[q] = p; } else { match_[p] = -1; match_[q] = -1; }
}
static int feasible_pair(int p, int q) {
    int a = p < q ? p : q, b = p < q ? q : p;
    for (int i = 0; i < 8; i++) if (K3rem[contrib[a][b][i][0]][contrib[a][b][i][1]] < 1) return 0;
    return 1;
}
static int all_nonneg(void) {
    for (int v = 0; v < nl; v++) for (int w = 0; w < nl; w++) if (K3rem[v][w] < 0) return 0;
    return 1;
}
static void rec(void) {
    if (stop_) return;
    nodes++;
    if (time_limit > 0 && (nodes & 1023) == 0 && (double)(clock() - t0) / CLOCKS_PER_SEC > time_limit) { stop_ = 1; return; }
    int best_v = -1, best_w = -1, best_b = 0; long long best_cost = 1LL << 60;
    static int best_opts[MAXNP * MAXNP][2]; int best_n = 0;
    int anybudget = 0;
    for (int v = 0; v < nl; v++) for (int w = 0; w < nl; w++) {
        int b = K3rem[v][w];
        if (b <= 0) continue;
        anybudget = 1;
        int t = pi_[w];
        int opts[MAXNP * 2][2], n = 0;
        for (int i = 0; i < nl - 2; i++) {
            int p = pairs_with[v][i];
            if (match_[p] >= 0) continue;
            for (int j = 0; j < nl - 2; j++) {
                int q = pairs_with[t][j];
                if (q == p || match_[q] >= 0) continue;
                if (feasible_pair(p, q)) { opts[n][0] = p; opts[n][1] = q; n++; }
            }
        }
        long long cost = (b == 1) ? n : (long long)n * (n - 1) / 2;
        if (b == 2 && n < 2) cost = 0;
        if (cost < best_cost) { best_cost = cost; best_v = v; best_w = w; best_b = b; best_n = n; memcpy(best_opts, opts, n * 2 * sizeof(int)); if (cost == 0) return; }
    }
    if (!anybudget) {
        for (int p = 0; p < np; p++) if (match_[p] < 0) return;   /* cannot happen: budgets track matched pairs */
        n_iota++;
        if (dump) { for (int p = 0; p < np; p++) fprintf(dump, "%d ", match_[p]); fprintf(dump, "\n"); }
        if (verbose) { printf("  iota #%lld at node %lld (%.1fs)\n", n_iota, nodes, (double)(clock() - t0) / CLOCKS_PER_SEC); fflush(stdout); }
        if (max_iota && n_iota >= max_iota) stop_ = 2;
        return;
    }
    int local_opts[MAXNP * MAXNP][2];
    memcpy(local_opts, best_opts, best_n * 2 * sizeof(int));
    if (best_b == 1) {
        for (int i = 0; i < best_n; i++) {
            int p = local_opts[i][0], q = local_opts[i][1];
            apply(p, q, 1);
            rec();
            apply(p, q, -1);
            if (stop_) return;
        }
    } else {
        for (int i = 0; i < best_n; i++) {
            int p = local_opts[i][0], q = local_opts[i][1];
            apply(p, q, 1);
            for (int j = i + 1; j < best_n; j++) {
                int p2 = local_opts[j][0], q2 = local_opts[j][1];
                if (p2 == p || p2 == q || q2 == p || q2 == q) continue;
                if (!feasible_pair(p2, q2)) continue;
                apply(p2, q2, 1);
                rec();
                apply(p2, q2, -1);
                if (stop_) { apply(p, q, -1); return; }
            }
            apply(p, q, -1);
        }
    }
}

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("open"); return 1; }
    if (fscanf(f, "%d", &m_) != 1) return 2;
    nl = 2 * (m_ - 1);
    for (int i = 0; i < nl; i++) if (fscanf(f, "%d", &mu[i]) != 1) return 2;
    for (int i = 0; i < nl; i++) if (fscanf(f, "%d", &tau[i]) != 1) return 2;
    fclose(f);
    if (argc > 2) time_limit = atof(argv[2]);
    if (argc > 3) max_iota = atoll(argv[3]);
    if (argc > 4) verbose = atoi(argv[4]);
    if (argc > 5) dump = fopen(argv[5], "w");
    for (int v = 0; v < nl; v++) { pi_[v] = v ^ 1; line_[v] = v / 2 + 1; }
    for (int v = 0; v < nl; v++) hh[v] = pi_[tau[v]];
    np = 0;
    for (int v = 0; v < nl; v++) for (int w = v + 1; w < nl; w++) if (mu[v] != w) { pair_v[np][0] = v; pair_v[np][1] = w; np++; }
    for (int v = 0; v < nl; v++) { int c = 0; for (int p = 0; p < np; p++) if (in_pair(p, v)) pairs_with[v][c++] = p; }
    int mup[MAXNL]; for (int w = 0; w < nl; w++) mup[w] = pi_[mu[pi_[w]]];
    for (int v = 0; v < nl; v++) for (int w = 0; w < nl; w++)
        K3[v][w] = (hh[v] == w ? 1 : 2) - (v == w) - (hh[mu[v]] == w) - (hh[v] == mup[w]);
    int neg = 0; for (int v = 0; v < nl; v++) for (int w = 0; w < nl; w++) if (K3[v][w] < 0) neg = 1;
    if (neg) { printf("iota2: invalid type (K3 < 0)\n"); return 0; }
    contrib = malloc(sizeof(int) * MAXNP * MAXNP * 16);
    for (int p = 0; p < np; p++) for (int q = 0; q < np; q++) {
        int v1 = pair_v[p][0], v2 = pair_v[p][1], w1 = pair_v[q][0], w2 = pair_v[q][1];
        int c[8][2] = {{v1, pi_[w1]}, {v1, pi_[w2]}, {v2, pi_[w1]}, {v2, pi_[w2]}, {w1, pi_[v1]}, {w1, pi_[v2]}, {w2, pi_[v1]}, {w2, pi_[v2]}};
        memcpy(contrib[p][q], c, sizeof(c));
    }
    memcpy(K3rem, K3, sizeof(K3));
    for (int p = 0; p < np; p++) match_[p] = -1;
    t0 = clock();
    rec();
    printf("iota2: %lld iota, nodes %lld, %.1fs, %s\n", n_iota, nodes, (double)(clock() - t0) / CLOCKS_PER_SEC, stop_ == 1 ? "STOPPED (time)" : (stop_ == 2 ? "STOPPED (max)" : "COMPLETE"));
    if (dump) fclose(dump);
    return 0;
}
