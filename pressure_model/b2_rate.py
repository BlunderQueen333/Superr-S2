"""Bowling stage 2 (v3). Same framework as the batters (m4_rate.py): impact = usual level + shrunk pressure deviation, in runs; runs saved plus WICKETS VALUED IN RUNS
(a wicket is worth the batter's expected future runs), shrunk separately and added.
Scored pillars: impact (70%), consistency (15%), discipline = wides / no-balls avoided (15%, capped: a general bowler trait). Wickets are inside impact; win impact and slope are
information only. Published rank = ensemble of 5 variants; rank range / top-10 chance = bootstrap of the ensemble with fixed shrinkage hyper-parameters.
"""
import sys, pickle, json, os
PYLIB, IN, DATA, OUT = sys.argv[1:5]
if PYLIB != '-': sys.path.insert(0, PYLIB)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
d = pickle.load(open(IN, 'rb')); e = d['e'].reset_index(drop=True); m = d['m']; BT = d['BT']
teams = pd.read_csv(f'{DATA}/teams.csv'); tname = dict(zip(teams.team_id, teams.team_name))
e['bowler'] = pd.to_numeric(e.bowler_id, errors='coerce'); e = e[e.bowler.notna()].reset_index(drop=True)
bowlers = sorted(e.bowler.unique()); bidx = {b: i for i, b in enumerate(bowlers)}; midx = {x: i for i, x in enumerate(sorted(e.match_id.unique()))}
NB, NM = len(bowlers), len(midx)
bi = e.bowler.map(bidx).astype(int).values; mi = e.match_id.map(midx).astype(int).values
w_imp = (0.5 + e.imp.values); w_imp = w_imp / w_imp.mean(); w_one = np.ones(len(e))
e['ig'] = e.groupby(['match_id', 'innings', 'bowler'], sort=False).ngroup(); ig = e.ig.values; NI = int(ig.max()) + 1
mp = pd.read_csv(f'{DATA}/competition_team_player_mapping.csv', keep_default_na=False, na_values=[]); ipl26 = set(mp[mp.comp_name == 'IPL 2026'].player_id)
pe = pd.read_csv(f'{DATA}/playing_eleven.csv', keep_default_na=False, na_values=[]); pe = pe[pe.is_played.astype(str).str.lower() == 'true']
pe = pe.merge(m[['match_id', 'match_date', 'match_format']], on='match_id'); last_played = pe.groupby('player_id').match_date.max()
intl26 = set(pe[(pe.match_format == 'T20I') & (pe.match_date >= '2026-01-01')].player_id)
active_b = np.array([(int(b) in ipl26) or (int(b) in intl26) for b in bowlers])
if os.environ.get('ACTIVE_ONLY', '0') != '1': active_b = np.ones(len(active_b), bool)          # default: EVERY player 2022-2026; ACTIVE_ONLY=1 restricts to IPL 2026 squads / 2026 T20I players
lg = e[e.legal == 1]; dsh = lg.groupby('bowler').over.apply(lambda x: (x >= 15).mean()).reindex(bowlers).fillna(0).values; psh = lg.groupby('bowler').over.apply(lambda x: (x <= 5).mean()).reindex(bowlers).fillna(0).values
role_b = np.where(dsh >= 0.30, 2, np.where(psh >= 0.45, 0, 1)); RN = ['Powerplay bowler (overs 1–6)', 'Middle-overs bowler (7–15)', 'Death bowler (16–20)']
# ---- per-delivery residuals, positive = better for the bowler
L = e.legal.values.astype(float)
R = {'wdc': e.wdc.values, 'econ': (e.ech - e.ch).values, 'wkt': (e.cw - e.pw).values, 'win': (e.bwpa - e.ebw).values, 'xt': (e.pxt - e.xt).values,
     'dot': np.where(L > 0, (e.dotb - e.pd).values, 0.0), 'bnd': np.where(L > 0, (e.pbnd - e.bnd).values, 0.0)}
X = e.bpix.values - 0.5; STR = e.strong_bat.values.astype(bool)
def mat(v, rows=bi, cols=mi):
    A = np.zeros((NB, NM)); np.add.at(A, (rows, cols), v); return A
def build(press, w):
    pm = press.astype(bool); M = {}; wp = w * pm; wl = wp * L
    M['W'] = mat(wp); M['W2'] = mat(w * w * pm); M['WL'] = mat(wl); M['WL2'] = mat(w * w * pm * L); M['n'] = mat(pm * L)
    for k in ('econ', 'wkt', 'win', 'xt', 'wdc'): M['P' + k] = mat(wp * R[k])
    calm = ~pm; M['nW'] = mat(w * calm); M['nW2'] = mat(w * w * calm)              # NORMAL balls: the bowler's usual level
    for k in ('econ', 'wdc'): M['N' + k] = mat(w * calm * R[k])
    for k in ('dot', 'bnd'): M['P' + k] = mat(wl * R[k])
    for k in ('win', 'econ'):
        M['aW' + k] = mat(w); M['aX' + k] = mat(w * X); M['aXX' + k] = mat(w * X * X); M['aR' + k] = mat(w * R[k]); M['aXR' + k] = mat(w * X * R[k])
    M['uN'] = mat(np.ones(len(e))); M['uX'] = mat(X); M['uXX'] = mat(X * X)
    M['sW'] = mat(wp * STR); M['sW2'] = mat(w * w * pm * STR); M['sR'] = mat(wp * STR * R['econ'])
    pb = np.bincount(ig, weights=pm * L, minlength=NI); ws = np.bincount(ig, weights=R['win'] * pm, minlength=NI)
    first = pd.Series(np.arange(len(e))).groupby(ig).first().values
    inn = (pb >= 6).astype(float); hit = inn * (ws >= 0); flo = inn * np.minimum(ws, 0)
    for nm_, arr in (('inn', inn), ('hit', hit), ('flo', flo)): M[nm_] = mat(arr[ig[first]], bi[first], mi[first])
    sig = {}
    for k in ('econ', 'wkt', 'win', 'xt', 'wdc'): mu = (wp * R[k]).sum() / wp.sum(); sig[k] = float((wp * (R[k] - mu) ** 2).sum() / wp.sum())
    for k in ('econ', 'wdc'): wn = w * calm; mu = (wn * R[k]).sum() / wn.sum(); sig[k + '_n'] = float((wn * (R[k] - mu) ** 2).sum() / wn.sum())
    for k in ('dot', 'bnd'): mu = (wl * R[k]).sum() / wl.sum(); sig[k] = float((wl * (R[k] - mu) ** 2).sum() / wl.sum())
    sig['slope_win'] = float(R['win'].var()); sig['slope_econ'] = float(R['econ'].var())
    pv = ws[inn > 0]; sig['hit_p'] = float((pv >= 0).mean()); sig['floor'] = float(np.minimum(pv, 0).var())
    return M, sig
TAUD = 0.0          # floor on the between-player spread of the pressure deviation (0 = data decides; >0 = pressure-tilted variant)
def eb(x, v, ok, tmin=0.0, fixed=None):
    """DerSimonian-Laird empirical Bayes; fixed=(tau2, mu) reuses hyper-parameters (bootstrap: duplicated matches inflate the apparent spread of players)"""
    good = ok & np.isfinite(x) & np.isfinite(v) & (v > 0)
    xs, vs = x[good], v[good]
    if fixed is None:
        wi = 1 / vs; mu0 = (wi * xs).sum() / wi.sum(); Q = (wi * (xs - mu0) ** 2).sum()
        tau2 = max(tmin ** 2, (Q - (len(xs) - 1)) / (wi.sum() - (wi ** 2).sum() / wi.sum())); mu = ((1 / (tau2 + vs)) * xs).sum() / (1 / (tau2 + vs)).sum()
    else: tau2, mu = fixed
    pm_ = np.zeros(len(x)); ps = np.full(len(x), np.nan); Bi = np.zeros(len(x))
    Bi[good] = tau2 / (tau2 + v[good]); pm_[good] = Bi[good] * (x[good] - mu); ps[good] = np.sqrt(Bi[good] * v[good])
    return pm_, ps, np.sqrt(tau2), Bi, (tau2, mu)
CK = ['econ', 'dot', 'bnd', 'strong', 'wkt', 'win', 'wdc', 'slope_win', 'slope_econ', 'hit', 'floor', 'xt', 'econ_n', 'wdc_n', 'econ_d', 'wdc_d', 'imph']
CN = {'wdc': 'Value of wickets in runs (batter\'s expected future runs)', 'econ_n': 'Usual runs saved (calm balls)', 'wdc_n': 'Usual wicket value (calm balls)', 'econ_d': 'Runs saved: pressure minus usual (shrunk)', 'wdc_d': 'Wicket value: pressure minus usual (shrunk)', 'imph': 'Impact under pressure = usual level + shrunk pressure deviation (runs saved + wickets valued in runs)', 'econ': 'Runs saved vs expected', 'dot': 'Dot balls above expected', 'bnd': 'Boundaries prevented vs expected', 'strong': 'Runs saved vs strong batting sides', 'wkt': 'Wickets above expected', 'win': 'Win probability added above expected',
      'slope_win': 'Pressure slope (win impact)', 'slope_econ': 'Pressure slope (runs saved)', 'hit': 'Share of pressure spells with positive win impact', 'floor': 'Size of the bad pressure spells', 'xt': 'Wides / no-balls avoided vs expected'}
PK = ['runs', 'wickets', 'win', 'slope', 'consistency', 'discipline']
PNAME = {'runs': 'Impact under pressure: runs saved plus wickets valued in runs (usual level + shrunk pressure deviation)', 'wickets': 'Wicket-taking above expected', 'win': 'Win impact above expected', 'slope': 'Pressure slope (rises with pressure vs own level)',
         'consistency': 'Consistency (hit rate and size of the bad spells)', 'discipline': 'Discipline under pressure (wides / no-balls)'}
PMEM = {'runs': ['imph'], 'wickets': ['wkt'], 'win': ['win'], 'slope': ['slope_win', 'slope_econ'], 'consistency': ['hit', 'floor'], 'discipline': ['xt']}
G = np.zeros((len(PK), len(CK)))
for a_, p_ in enumerate(PK):
    for k_ in PMEM[p_]: G[a_, CK.index(k_)] = 1 / len(PMEM[p_])

def estimates(T, sig):
    o = {}; W = np.maximum(T['W'], 1e-9); WL = np.maximum(T['WL'], 1e-9)
    for k in ('econ', 'wkt', 'win', 'xt', 'wdc'): o[k] = (np.where(T['W'] > 0, T['P' + k] / W, np.nan), sig[k] * T['W2'] / W ** 2)
    Wn = np.maximum(T['nW'], 1e-9)
    for k in ('econ', 'wdc'):
        xn = np.where(T['nW'] > 0, T['N' + k] / Wn, np.nan); vn = sig[k + '_n'] * T['nW2'] / Wn ** 2; o[k + '_n'] = (xn, vn); o[k + '_d'] = (o[k][0] - xn, o[k][1] + vn)
    for k in ('dot', 'bnd'): o[k] = (np.where(T['WL'] > 0, T['P' + k] / WL, np.nan), sig[k] * T['WL2'] / WL ** 2)
    sW = np.maximum(T['sW'], 1e-9); o['strong'] = (np.where(T['sW'] > 0, T['sR'] / sW, np.nan), sig['econ'] * T['sW2'] / sW ** 2)
    for k in ('win', 'econ'):
        sw = np.maximum(T['aW' + k], 1e-9); sxx_w = T['aXX' + k] - T['aX' + k] ** 2 / sw; sxr = T['aXR' + k] - T['aX' + k] * T['aR' + k] / sw; sxx_u = T['uXX'] - T['uX'] ** 2 / np.maximum(T['uN'], 1)
        o['slope_' + k] = (np.where(sxx_w > 1e-6, sxr / np.maximum(sxx_w, 1e-9), np.nan), np.where(sxx_u > 1e-6, sig['slope_' + k] / np.maximum(sxx_u, 1e-9), np.nan))
    n = np.maximum(T['inn'], 1e-9); p = sig['hit_p']
    o['hit'] = (np.where(T['inn'] > 0, T['hit'] / n, np.nan), p * (1 - p) / n); o['floor'] = (np.where(T['inn'] > 0, T['flo'] / n, np.nan), sig['floor'] / n)
    return o
def components(T, sig, ok, fixed=None, rec=None):
    est = estimates(T, sig); Z = np.zeros((len(CK), NB)); S = np.full((len(CK), NB), np.nan); tau = np.zeros(len(CK)); B = np.zeros((len(CK), NB)); PM = {}
    COMB = {'imph': ('econ_n', 'wdc_n', 'econ_d', 'wdc_d')}
    for j, k in enumerate(CK):
        if k in COMB: continue
        x, v = est[k]; vv = fixed[k][2] if fixed else v
        pm_, ps, t, Bi, hp = eb(x, vv, ok, TAUD if k in ('econ_d', 'wdc_d') else 0.0, (fixed[k][0], fixed[k][1]) if fixed else None); tau[j] = t; B[j] = Bi; PM[k] = (pm_, ps)
        if rec is not None: rec[k] = (hp[0], hp[1], v)
        if t > 1e-12: Z[j] = pm_ / t; S[j] = ps / t
    for k, mem in COMB.items():
        j = CK.index(k); pm_i = sum(PM[x][0] for x in mem); sd_i = np.sqrt(sum(np.nan_to_num(PM[x][1], nan=0.0) ** 2 for x in mem))
        good = (fixed[k + '_good'] if fixed else ok & (B[CK.index(mem[0])] > 0))
        t = fixed[k + '_t'] if fixed else (pm_i[good].std() if good.sum() > 5 else 0.0); tau[j] = t; B[j] = np.mean([B[CK.index(x)] for x in mem], axis=0)
        if rec is not None: rec[k + '_t'] = t; rec[k + '_good'] = good
        if t > 1e-12: Z[j] = pm_i / t; S[j] = np.where(good, sd_i / t, np.nan)
    return Z, S, tau, est, B
def raw_z(x, ok):
    out = np.full(NB, np.nan); g = ok & np.isfinite(x); out[g] = np.clip((x[g] - x[g].mean()) / (x[g].std() + 1e-12), -3, 3); return np.nan_to_num(out)
def target_Y(Tb, sig, ok):
    est = estimates(Tb, sig); return 0.5 * (raw_z(est['econ'][0] + est['wdc'][0], ok) + raw_z(est['win'][0], ok))
def spear(a, b, ok):
    idx = np.where(ok)[0]
    if np.std(a[idx]) < 1e-12 or np.std(b[idx]) < 1e-12: return 0.0
    return float(spearmanr(a[idx], b[idx])[0])
press0 = e.press.values == 1
M0, SIG0 = build(press0, w_imp); Tfull = {k: v.sum(1) for k, v in M0.items()}
NP_MIN, PINN_MIN = 150, 12
ok_pop = Tfull['n'] >= 40; elig = (Tfull['n'] >= NP_MIN) & (Tfull['inn'] >= PINN_MIN) & active_b; elig_all = (Tfull['n'] >= NP_MIN) & (Tfull['inn'] >= PINN_MIN)
print('bowlers', NB, '| pressure deliveries', int(press0.sum()), '| with >=40 pressure balls', int(ok_pop.sum()), '| eligible (active)', int(elig.sum()), {RN[k]: int((role_b[elig] == k).sum()) for k in range(3)})
Z0, S0, TAU0, EST0, B0 = components(Tfull, SIG0, ok_pop); print('tau:', dict(zip(CK, np.round(TAU0, 4))))
def halves(sel, mask=None):
    mk = np.ones(NM, bool) if mask is None else mask
    Ta = {k: v[:, sel & mk].sum(1) for k, v in M0.items()}; Tb = {k: v[:, ~sel & mk].sum(1) for k, v in M0.items()}
    return Ta, Tb, (Ta['n'] >= 30) & (Tb['n'] >= 30)
def run_halves(seeds, mask=None):
    rep = np.zeros(len(CK)); val = np.zeros(len(CK)); prep = np.zeros(len(PK)); pval = np.zeros(len(PK))
    for s_ in seeds:
        sel = np.random.default_rng(s_).random(NM) < .5; Ta, Tb, ok_ = halves(sel, mask)
        Za, _, _, _, _ = components(Ta, SIG0, ok_); Zb, _, _, _, _ = components(Tb, SIG0, ok_); Y = target_Y(Tb, SIG0, ok_); Pa = G @ Za; Pb = G @ Zb
        for j in range(len(CK)): rep[j] += spear(Za[j], Zb[j], ok_) / len(seeds); val[j] += spear(Za[j], Y, ok_) / len(seeds)
        for j in range(len(PK)): prep[j] += spear(Pa[j], Pb[j], ok_) / len(seeds); pval[j] += spear(Pa[j], Y, ok_) / len(seeds)
    return rep, val, prep, pval
rep_fit, val_fit, prep_fit, pval_fit = run_halves(range(100, 130)); rep_rep, val_rep, prep_rep, pval_rep = run_halves(range(300, 330))
print('component repeat (report):', dict(zip(CK, np.round(rep_rep, 2)))); print('component validity (report):', dict(zip(CK, np.round(val_rep, 2))))
print('PILLAR repeat (fit)', dict(zip(PK, np.round(prep_fit, 2))), '| validity (fit)', dict(zip(PK, np.round(pval_fit, 2)))); print('PILLAR repeat (report)', dict(zip(PK, np.round(prep_rep, 2))), '| validity (report)', dict(zip(PK, np.round(pval_rep, 2))))
WFIT = None
# CHOSEN WEIGHTS: only pillars with a rationale AND some signal are scored (wickets are already inside impact; win-impact and slope showed no persistent signal -> information only)
# Impact 70% / Consistency 15% / Discipline 15% (discipline repeats well but is a general bowler trait, so it is capped at 15%)
WV = np.array([0.70, 0.0, 0.0, 0.0, 0.15, 0.15]); WE = np.array([0.5, 0.0, 0.0, 0.0, 0.25, 0.25])
print('chosen pillar weights:', dict(zip(PK, WV)))
LEVEL = np.array([1, 0, 0, 0, 0, 0], float)
def design_scores(Za, Wv): Pa = G @ Za; return {'chosen weights': (Wv[:, None] * Pa).sum(0), 'impact 50 / consistency 25 / discipline 25': (WE[:, None] * Pa).sum(0), 'impact only': (LEVEL[:, None] * Pa).sum(0)}
pred = {}
for s_ in range(300, 330):
    sel = np.random.default_rng(s_).random(NM) < .5; Ta, Tb, ok_ = halves(sel); Za, _, _, _, _ = components(Ta, SIG0, ok_); Zb, _, _, _, _ = components(Tb, SIG0, ok_); Y = target_Y(Tb, SIG0, ok_)
    for k_, v_ in design_scores(Za, WV).items(): pred.setdefault(k_, []).append(spear(v_, Y, ok_))
    pred.setdefault('_comp_repeat', []).append(spear((WV[:, None] * (G @ Za)).sum(0), (WV[:, None] * (G @ Zb)).sum(0), ok_))
dts = e.groupby('match_id').match_date.first(); era = np.array([dts[x][:4] >= '2024' for x in sorted(e.match_id.unique())]); era_res = {}
for lab, (fit_m, test_m) in {'2022-23 → 2024-26': (~era, era), '2024-26 → 2022-23': (era, ~era)}.items():
    Ta = {k: v[:, fit_m].sum(1) for k, v in M0.items()}; Tb = {k: v[:, test_m].sum(1) for k, v in M0.items()}; ok_ = (Ta['n'] >= 30) & (Tb['n'] >= 30); Za, _, _, _, _ = components(Ta, SIG0, ok_)
    vv = np.zeros(len(PK)); rr_ = np.zeros(len(PK))
    for s_ in range(500, 515):
        sel = np.random.default_rng(s_).random(NM) < .5; Ta2, Tb2, ok2 = halves(sel, fit_m); Za2, _, _, _, _ = components(Ta2, SIG0, ok2); Zb2, _, _, _, _ = components(Tb2, SIG0, ok2); Y2 = target_Y(Tb2, SIG0, ok2); P2 = G @ Za2; Pb2 = G @ Zb2
        for j in range(len(PK)): vv[j] += spear(P2[j], Y2, ok2) / 15; rr_[j] += spear(P2[j], Pb2[j], ok2) / 15
    Wf = WV; Y = target_Y(Tb, SIG0, ok_); era_res[lab] = {k_: round(spear(v_, Y, ok_), 3) for k_, v_ in design_scores(Za, Wf).items()}; era_res[lab]['n'] = int(ok_.sum())
print('predictive validity:', {k: round(float(np.mean(v)), 3) for k, v in pred.items()}); print('chronological:', era_res)
def composite(T, sig, W, fixed=None, rec=None): Z, S, _, _, _ = components(T, sig, ok_pop, fixed, rec); return (W[:, None] * (G @ Z)).sum(0), Z
comp0, Zc = composite(Tfull, SIG0, WV); eidx = np.where(elig)[0]; mu_c = comp0[eidx].mean(); sd_c = comp0[eidx].std()
# ---- ENSEMBLE over defensible analyst choices; the bootstrap is run on the ENSEMBLE with shrinkage hyper-parameters held fixed (duplicated matches would otherwise inflate the apparent spread of bowlers)
rng = np.random.default_rng(7); BR = 300; DRAWS = np.stack([np.bincount(rng.integers(0, NM, NM), minlength=NM).astype(float) for _ in range(BR)])
q = lambda p: np.quantile(e.bpix.values, p)
ENS = [('Base definition (data decides how much pressure deviation counts)', press0, w_imp, WV, 0.0), ('Pressure = top 30% of the continuous pressure level', e.bpix.values >= q(0.70), w_imp, WV, 0.0),
       ('Pressure = top 15% of the continuous pressure level', e.bpix.values >= q(0.85), w_imp, WV, 0.0), ('No match-importance weighting', press0, w_one, WV, 0.0),
       ('Pressure-tilted: allow a modest real pressure effect (floor 0.04 runs/ball)', press0, w_imp, WV, 0.04)]
def zs(c): x = c[eidx]; return (x - x.mean()) / x.std()
ens_pt, ens_bt, sens = [], [], {}
for name, pr, w, W, taud in ENS:
    TAUD = taud; Mx, sx = (M0, SIG0) if name.startswith('Base') else build(pr, w)
    Tx = {k: v.sum(1) for k, v in Mx.items()}; REC = {}; c, Zx = composite(Tx, sx, W, rec=REC); zp = zs(c); ens_pt.append(zp); sens[name] = {'z': zp, 'ens': True}
    zb = np.zeros((BR, len(eidx)))
    for r in range(BR): cb, _ = composite({k: v @ DRAWS[r] for k, v in Mx.items()}, sx, W, fixed=REC); zb[r] = zs(cb)
    ens_bt.append(zb)
    if not name.startswith('Base'): del Mx
TAUD = 0.0
ens_z = np.mean(ens_pt, axis=0); ENB = np.mean(ens_bt, axis=0); ens_s = (ens_z - ens_z.mean()) / ens_z.std()
ranks = np.zeros((BR, len(eidx)), dtype=int)
for r in range(BR): ranks[r][np.argsort(-ENB[r])] = np.arange(1, len(eidx) + 1)
med_rank = np.median(ranks, axis=0); top10p = (ranks <= 10).mean(0); rlo = np.percentile(ranks, 5, axis=0); rhi = np.percentile(ranks, 95, axis=0)
order_cons = np.argsort(-ens_z); final_rank = {int(eidx[i]): k + 1 for k, i in enumerate(order_cons)}; ens_score = {int(eidx[i]): float(ens_s[i]) for i in range(len(eidx))}
base10 = set(order_cons[:10]); base30 = set(order_cons[:30])
def addsens(name, press=press0, w=w_imp, Wf=None):
    Mx, sx = build(press, w); Tx = {k: v.sum(1) for k, v in Mx.items()}; c, _ = composite(Tx, sx, WV if Wf is None else Wf); sens[name] = {'z': zs(c), 'ens': False}
addsens('Pressure = Bowling Pressure Index only (top 15%)', press=(e.bpi.values >= e.bpi.quantile(0.85)))
for j, k in enumerate(PK):
    if WV[j] > 0: wk = WV.copy(); wk[j] = 0; addsens('Without pillar: ' + PNAME[k], Wf=wk / wk.sum())
for nm_, wv_ in (('Impact only', [1, 0, 0, 0, 0, 0]), ('Impact 50 / consistency 25 / discipline 25', [0.5, 0, 0, 0, 0.25, 0.25]), ('Impact 60 / consistency 20 / discipline 20', [0.6, 0, 0, 0, 0.2, 0.2])): addsens(nm_, Wf=np.array(wv_, float))
for v in sens.values(): o = np.argsort(-v['z']); v['o10'] = len(base10 & set(o[:10])); v['o30'] = len(base30 & set(o[:30]))
sens = {k: {'o10': v['o10'], 'o30': v['o30'], 'ens': v['ens']} for k, v in sens.items()}
print('sensitivity overlap with final top 10 / top 30:', {k: (v['o10'], v['o30']) for k, v in sens.items()})
# ---- records
nm_b = e.groupby('bowler').bowler_name.first(); type_b = e.groupby('bowler').pace_or_spin.agg(lambda s: s.value_counts().index[0]); btyp_b = e.groupby('bowler').bowling_type.agg(lambda s: s.value_counts().index[0])
team_b = e.groupby(['bowler', 'bowling_team_id']).size().reset_index(name='n').sort_values(['bowler', 'n'], ascending=[True, False])
pos_all = {int(bowlers[i]): k for k, i in enumerate(np.argsort(-comp0))}
def st(df):
    lb = float(df.legal.sum())
    if lb == 0: return None
    wk = int(df.cw.sum()); ch = float(df.ch.sum())
    return {'balls': int(lb), 'runs': int(ch), 'wkts': wk, 'econ': round(6 * ch / lb, 2), 'eecon': round(6 * float(df.ech.sum()) / lb, 2), 'dot': round(100 * float(df.dotb.sum()) / lb, 1), 'edot': round(100 * float(df.pd.sum()) / lb, 1),
            'bnd': round(100 * float(df.bnd.sum()) / lb, 1), 'ebnd': round(100 * float(df.pbnd.sum()) / lb, 1), 'ewk': round(float(df.pw.sum()), 1), 'xt': int(df.xt.sum()), 'exp_xt': round(float(df.pxt.sum()), 1), 'win': round(float((df.bwpa - df.ebw).sum()), 2)}
sp_tab = e[e.press == 1].groupby('ig').agg(date=('match_date', 'first'), bat=('batting_team_id', 'first'), balls=('legal', 'sum'), runs=('ch', 'sum'), wk=('cw', 'sum'), net=('bwpa', 'sum'), eb=('ebw', 'sum'), bowler=('bowler', 'first'))
sp_tab['gain'] = sp_tab.net - sp_tab.eb; sp_tab = sp_tab[sp_tab.balls >= 6]; spg = sp_tab.groupby('bowler')
eligpos = {int(eidx[i]): i for i in range(len(eidx))}; recs = []; eg = e.groupby('bowler')
for b in bowlers:
    i = bidx[b]; df = eg.get_group(b)
    if df.legal.sum() < 60: continue
    pr = df[df.press == 1]; nr = df[df.press == 0]; isel = i in eligpos; k = eligpos.get(i)
    r = {'id': int(b), 'name': str(nm_b.get(b, b)), 'teams': [tname.get(t, str(t)) for t in team_b[team_b.bowler == b].bowling_team_id.head(2)], 'role': RN[role_b[i]], 'type': str(type_b.get(b, '')), 'btype': str(btyp_b.get(b, '')), 'balls': int(df.legal.sum()), 'elig': bool(isel),
         'active': bool(active_b[i]), 'last': str(last_played.get(int(b), ''))[:10], 'pn': int(Tfull['n'][i]), 'pinn': int(Tfull['inn'][i]), 'press': st(pr), 'norm': st(nr), 'ipl': round(float(df.is_ipl.mean()), 3), 'pshare': round(float((df.press * df.legal).sum() / max(df.legal.sum(), 1)), 3),
         'types': {BT[t]: st(pr[pr.ptype == t]) for t in range(len(BT)) if pr[pr.ptype == t].legal.sum() >= 18}, 'str': st(df[df.strong_bat]), 'wk': st(df[~df.strong_bat])}
    if isel or elig_all[i]:
        r.update({'score': round(50 + 10 * ens_score[i], 1) if i in ens_score else round(50 + 10 * (float(comp0[i]) - mu_c) / sd_c, 1), 'score_base': round(50 + 10 * (float(comp0[i]) - mu_c) / sd_c, 1), 'rank_all': pos_all[int(b)] + 1, 'impS': round(6 * float(Zc[CK.index('imph')][i]) * float(TAU0[CK.index('imph')]), 2),
                  'z': [round(float(Zc[j][i]), 2) for j in range(len(CK))], 'zp': [round(float((G @ Zc)[j][i]), 2) for j in range(len(PK))], 'hit': round(float(Tfull['hit'][i] / max(Tfull['inn'][i], 1)), 2)})
    if isel:
        sdv = float(np.sqrt(np.nansum((WV * np.sqrt(np.maximum(G ** 2 @ np.nan_to_num(S0[:, i], nan=0.0) ** 2, 0))) ** 2)))
        best = spg.get_group(b).sort_values('gain', ascending=False).head(3) if b in spg.groups else []
        r.update({'rank': final_rank[i], 'medrank': float(med_rank[k]), 'rlo': int(rlo[k]), 'rhi': int(rhi[k]), 'top10p': round(float(top10p[k]), 3), 'sd': round(sdv / sd_c * 10, 2),
                  'best': [[x.date, tname.get(x.bat, ''), int(x.wk), int(x.runs), int(x.balls)] for x in best.itertuples()] if len(best) else []})
    recs.append(r)
cal = lambda g: [[str(i), int(x.legal.sum()), round(6 * x.ch.sum() / max(x.legal.sum(), 1), 2), round(6 * x.ech.sum() / max(x.legal.sum(), 1), 2), round(6 * x.cw.sum() / max(x.legal.sum(), 1), 3), round(6 * x.pw.sum() / max(x.legal.sum(), 1), 3)] for i, x in g]
calib = {'phase': cal(e.assign(p=np.where(e.over < 6, 'overs 1-6', np.where(e.over < 15, 'overs 7-15', 'overs 16-20'))).groupby('p')), 'situation': cal(e.assign(s=np.where(e.press == 1, 'pressure', 'other')).groupby('s')), 'type': cal(e.assign(t=np.where(e.pace_or_spin == 'Pace', 'pace', 'spin')).groupby('t')), 'innings': cal(e.groupby('innings'))}
print('bowling calibration (balls, econ actual/expected, wkts per over actual/expected):', {k: [(r[0], r[2], r[3], r[4], r[5]) for r in v] for k, v in calib.items()})
meta = {'calib': calib, 'matches': int(e.match_id.nunique()), 'deliveries': int(len(e)), 'bowlers': NB, 'eligible': int(elig.sum()), 'pressShare': round(float(press0.mean()), 3), 'BT': BT, 'BPIW': d['BPIW'], 'minPress': NP_MIN, 'minSpells': PINN_MIN, 'roles': {RN[k]: int((role_b[elig] == k).sum()) for k in range(3)},
        'pressureByOver': [[o, round(float(e[e.over == o].press.mean()), 3)] for o in range(20)], 'activeRule': 'IPL 2026 squad or a T20I in 2026',
        'pillars': {'keys': PK, 'names': [PNAME[k] for k in PK], 'weights': [round(float(x), 4) for x in WV], 'repeat': [round(float(x), 2) for x in prep_rep], 'validity': [round(float(x), 2) for x in pval_rep], 'members': PMEM},
        'comp': {'keys': CK, 'names': [CN[k] for k in CK], 'tau': [round(float(x), 5) for x in TAU0], 'repeat': [round(float(x), 2) for x in rep_rep], 'validity': [round(float(x), 2) for x in val_rep], 'pred': {k: round(float(np.mean(v)), 3) for k, v in pred.items()}, 'era': era_res}}
json.dump({'meta': meta, 'players': recs, 'sens': sens}, open(OUT, 'w'), separators=(',', ':'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))
tops = sorted([r for r in recs if r['elig']], key=lambda r: r['rank'])
print('\nFINAL bowlers')
for r in tops[:30]: print(r['rank'], r['name'], '|', r['role'][:6], r['type'], '| score', r['score'], '±', r['sd'], '| medrank', r['medrank'], '[%d-%d]' % (r['rlo'], r['rhi']), 'top10p', r['top10p'], '| pn', r['pn'], 'pinn', r['pinn'], 'hit', r['hit'], '| pillars', r['zp'])
print('roles in top 10:', {RN[k][:6]: sum(1 for r in tops[:10] if r['role'] == RN[k]) for k in range(3)}, '| pace/spin', {t: sum(1 for r in tops[:10] if r['type'] == t) for t in ('Pace', 'Spin')})
