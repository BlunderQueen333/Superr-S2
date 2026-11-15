"""v4 rating engine (batters AND bowlers).  usage: python v4_rate.py - bat|bowl stage_pickle DATA out.json

Score = fixed, a-priori weights over nine shrunk metrics (see BAT_W / BOWL_W).  Weights are NOT fitted: validity numbers in this data (Spearman ~0.0-0.1 on ~150 players,
standard error ~0.08) cannot distinguish one weighting from another.  Instead the weights are stress-tested (random +-35% perturbation x match bootstrap) and the final
top 10 is chosen by P(top-10) and ordered by median rank.

Every metric = estimate x_i with sampling variance v_i  ->  DerSimonian-Laird empirical-Bayes shrinkage  ->  posterior mean / tau  (population-signal units, Z).
Composite = sum_k w_k Z_k.
"""
import sys, os, json, pickle
PYLIB, KIND, IN, DATA, OUT = sys.argv[1:6]
if PYLIB != '-': sys.path.insert(0, PYLIB)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
BAT = KIND == 'bat'
HERE = os.path.dirname(os.path.abspath(__file__))
BR = int(os.environ.get('BR', 400)); SIGW = 0.35
d = pickle.load(open(IN, 'rb')); m = d['m']
teams = pd.read_csv(f'{DATA}/teams.csv'); tname = dict(zip(teams.team_id, teams.team_name))
import official_strength as OS
_, _, tn_, _, _, s_ipl, s_icc = OS.build(m, teams, HERE)

# ------------------------------------------------------------------ weights (nominal, fixed)
BAT_W = {'imp': .28, 'wpa_cc': .10, 'big': .08, 'relief': .06, 'delta': .18, 'hit': .11, 'floor': .10, 'finish': .05, 'protect': .04}
BOWL_W = {'runs': .28, 'win_cl': .10, 'qwk': .08, 'creation': .06, 'delta': .18, 'hit': .11, 'floor': .10, 'bounce': .05, 'closing': .04}
BLOCK = {'imp': 'Level', 'wpa_cc': 'Level', 'big': 'Level', 'runs': 'Level', 'win_cl': 'Level', 'qwk': 'Level', 'relief': 'Relief', 'creation': 'Relief', 'delta': 'Delta',
         'hit': 'Reliability', 'floor': 'Reliability', 'finish': 'Reliability', 'bounce': 'Reliability', 'protect': 'Tail', 'closing': 'Tail'}
WN = BAT_W if BAT else BOWL_W
CK = list(WN); W0 = np.array([WN[k] for k in CK]); assert abs(W0.sum() - 1) < 1e-9
CN = ({'imp': 'Impact runs above expected, strict pressure (runs scored + cost of getting out)', 'wpa_cc': 'Win probability added above expected, chases and close finishes', 'big': 'Impact vs top-4 opposition and in knock-out / must-win games',
       'relief': 'Pressure relief: win probability added above expected in overs 1-6, non-pressure balls', 'delta': 'Pressure delta: pressure impact minus own non-pressure impact', 'hit': 'Share of pressure innings with positive impact',
       'floor': 'Downside: size of the bad pressure innings', 'finish': 'Chase finishing: not out in a won chase, last 4 overs, game still live', 'protect': 'Strike protection with a tail-ender partner (gated on runs above expected)'} if BAT else
      {'runs': 'Runs saved vs expected, strict pressure', 'win_cl': 'Win probability added defending the last 2 overs and close games', 'qwk': 'Quality-weighted wickets above expected', 'creation': 'Pressure creation: win probability added above expected in overs 1-6, non-pressure balls',
       'delta': 'Pressure delta: runs saved under pressure minus own non-pressure runs saved', 'hit': 'Share of pressure spells with positive win impact', 'floor': 'Downside: size of the bad pressure spells',
       'bounce': 'Bounce-back: runs saved + wicket value on the ball after conceding a boundary (game live)', 'closing': 'Tail clean-up: win impact against batting-order 8-11 while the game is live'})

# ------------------------------------------------------------------ rows
if BAT:
    r = d['bb'].reset_index(drop=True); r['pl'] = r.bat
    _bt = pd.to_numeric(r.bowling_team_id, errors='coerce').values
    r['strong'] = np.array([(s_ipl.get((c_, t_), 99) <= 4) if i_ == 1 else (s_icc.get((int(y_), tn_.get(t_)), 99) <= 4) for c_, t_, i_, y_ in zip(r.comp_id, _bt, r.is_ipl, r.year)])
    r['name'] = r.striker_name
    r['unit'] = r.groupby(['match_id', 'innings', 'bat'], sort=False).ngroup()
else:
    r = d['e'].sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True); r['pl'] = pd.to_numeric(r.bowler_id, errors='coerce'); r = r[r.pl.notna()].reset_index(drop=True)
    r['strong'] = r.strong_bat.astype(bool); r['name'] = r.bowler_name
    r['unit'] = r.groupby(['match_id', 'innings', 'pl'], sort=False).ngroup()
STRICT = r.press.values == 1; BROAD = r.press_broad.values == 1; TIER = os.environ.get('TIER', 'strict'); PRESS = STRICT if TIER == 'strict' else BROAD
players = np.array(sorted(r.pl.unique())); pidx = {p: i for i, p in enumerate(players)}; NP = len(players)
mids = sorted(r.match_id.unique()); midx = {x: i for i, x in enumerate(mids)}; NM = len(mids)
PI = r.pl.map(pidx).astype(int).values; MI = r.match_id.map(midx).astype(int).values
dts = r.groupby('match_id').match_date.first(); dmax = pd.Timestamp(m.match_date.max())
age_days = np.array([(dmax - pd.Timestamp(str(dts[x])[:10])).days for x in mids]); REC = 0.5 ** (age_days / (3 * 365.0))              # recency weight, 3-year half-life (per match)
wimp = (0.5 + r.imp.values); wimp = wimp / wimp.mean()
def row_w(use_imp=True, use_rec=True):
    w = wimp.copy() if use_imp else np.ones(len(r))
    if use_rec: w = w * REC[MI]
    return w / w.mean()
NU = int(r.unit.max()) + 1; UN = r.unit.values; FIRST = pd.Series(np.arange(len(r))).groupby(UN).first().values
live = r.wp.between(0.06, 0.94).values; close = (np.abs(r.wp.values - 0.5) < 0.25)
def mat(v, rows=PI, cols=MI):
    A = np.zeros((NP, NM)); np.add.at(A, (rows, cols), v); return A

# ------------------------------------------------------------------ per-row residuals (positive = better for the player)
if BAT:
    RAE, IAR, WAA = r.rae.values, r.iar.values, r.waa.values; DC = IAR - RAE
    INN2 = r.innings.values == 2; STRONG = r.strong.values; HIS = (r.imp.values >= 0.9)
    sp = d['sp'].copy(); sp['pi'] = sp.setb.map(pidx); sp['mi'] = sp.match_id.map(midx); sp = sp.dropna(subset=['pi', 'mi']); sp['res'] = sp.faced - sp.groupby('over').faced.transform('mean')
    SPI, SPM, SPR = sp.pi.astype(int).values, sp.mi.astype(int).values, sp.res.values; SPG = sp.rae_f.fillna(0).values; SPF = sp.faced.values
else:
    L = r.legal.values.astype(float)
    ECON = (r.ech - r.ch).values; WDC = r.wdc.values; WIN = (r.bwpa - r.ebw).values; WKT = (r.cw - r.pw).values
    sq = r.sq_r.fillna(0).values; QW = 1 + 0.25 * np.clip(sq / (sq.std() + 1e-9), -2, 2)
    bnd_prev = np.zeros(len(r), bool); g_ = (r.match_id.values[1:] == r.match_id.values[:-1]) & (r.innings.values[1:] == r.innings.values[:-1]) & (r.pl.values[1:] == r.pl.values[:-1]) & (r.bnd.values[:-1] == 1)
    bnd_prev[1:] = g_
    LASTTWO = (r.over.values >= 18); TAILB = (r.batting_order.values >= 8)
    BNC = ECON + WDC

# ------------------------------------------------------------------ additive parts: num, den, den2 matrices + residual variance
def part_mean(rv, mask, w, sig_from=None):
    mk = mask.astype(float); sel = mk > 0
    mu = (w[sel] * rv[sel]).sum() / max(w[sel].sum(), 1e-9); sig2 = float((w[sel] * (rv[sel] - mu) ** 2).sum() / max(w[sel].sum(), 1e-9))
    cl = mat(w * mk * (rv - mu)); return {'num': mat(w * mk * rv), 'den': mat(w * mk), 'den2': cl ** 2, 'iid': mat(w * w * mk), 'sig2': sig2}
def part_units(val, inn, sig2=None):
    """per-innings / per-spell units: val and the unit-existence flag inn (both per UNIT)"""
    n = mat(inn[UN[FIRST]], PI[FIRST], MI[FIRST]); s = mat(val[UN[FIRST]] * inn[UN[FIRST]], PI[FIRST], MI[FIRST])
    x = val[inn > 0]; mu = x.mean(); return {'num': s, 'den': n, 'den2': (s - mu * n) ** 2, 'iid': n, 'sig2': float(x.var()) if sig2 is None else sig2}
def build(press, w):
    P = {}; pm = press.astype(bool)
    if BAT:
        P['rae'] = part_mean(RAE, pm, w); P['dc'] = part_mean(DC, pm, w); P['imp_p'] = part_mean(IAR, pm, w); P['imp_n'] = part_mean(IAR, ~pm, w)
        P['wpa_cc'] = part_mean(WAA, pm & (INN2 | close), w); P['big'] = part_mean(IAR, pm & (STRONG | HIS), w); P['relief'] = part_mean(WAA, (~pm) & (r.lb.values < 36), w)
        pb = np.bincount(UN, weights=pm.astype(float), minlength=NU); ws = np.bincount(UN, weights=IAR * pm, minlength=NU); inn = (pb >= 6).astype(float)
        hit = (ws >= 0).astype(float); P['hit'] = part_units(hit, inn, sig2=None)
        p = hit[inn > 0].mean(); P['hit']['sig2'] = float(p * (1 - p)); P['floor'] = part_units(np.minimum(ws, 0), inn)
        late = (r.lb.values >= 96); cb = np.bincount(UN, weights=late.astype(float), minlength=NU); mw = np.bincount(UN, weights=late * r.wp.values, minlength=NU) / np.maximum(cb, 1)
        outs = np.bincount(UN, weights=(r.a_out.values == 1).astype(float), minlength=NU); won = np.bincount(UN, weights=r.bat_won.values.astype(float), minlength=NU) / np.bincount(UN, minlength=NU)
        fin = ((r.innings.groupby(UN).first().values == 2) & (cb >= 6) & (mw > 0.08) & (mw < 0.92)).astype(float); succ = ((won > 0.5) & (outs == 0)).astype(float)
        P['finish'] = part_units(succ, fin); pp = succ[fin > 0].mean(); P['finish']['sig2'] = float(pp * (1 - pp))
        _pc = mat(SPR - SPR.mean(), SPI, SPM); P['protect'] = {'num': mat(SPR, SPI, SPM), 'den': mat(np.ones(len(SPR)), SPI, SPM), 'den2': _pc ** 2, 'iid': mat(np.ones(len(SPR)), SPI, SPM), 'sig2': float(SPR.var())}
        P['gR'] = {'num': mat(SPG, SPI, SPM), 'den': mat(SPF, SPI, SPM), 'den2': np.zeros((NP, NM)), 'iid': np.zeros((NP, NM)), 'sig2': 1.0}
    else:
        P['runs'] = part_mean(ECON, pm, w); P['wdc'] = part_mean(WDC, pm, w); P['rn'] = part_mean(ECON, ~pm, w)
        P['win_cl'] = part_mean(WIN, pm & (LASTTWO | (np.abs(r.wp.values - 0.5) < 0.2)), w); P['qwk'] = part_mean(WKT * QW, pm, w)
        P['creation'] = part_mean(WIN, (~pm) & (r.over.values < 6), w); P['bounce'] = part_mean(BNC, bnd_prev & live, w); P['closing'] = part_mean(WIN, TAILB & live & (L > 0), w)
        pb = np.bincount(UN, weights=pm * L, minlength=NU); ws = np.bincount(UN, weights=WIN * pm, minlength=NU); inn = (pb >= 6).astype(float); hit = (ws >= 0).astype(float)
        P['hit'] = part_units(hit, inn); p = hit[inn > 0].mean(); P['hit']['sig2'] = float(p * (1 - p)); P['floor'] = part_units(np.minimum(ws, 0), inn)
    P['n'] = {'num': mat(pm.astype(float)), 'den': mat(pm.astype(float)), 'den2': np.zeros((NP, NM)), 'iid': np.zeros((NP, NM)), 'sig2': 1.0}
    P['inn'] = P['hit']
    return P
def totals(P, wv):
    out = {}
    for k, p in P.items():
        num = p['num'] @ wv; den = p['den'] @ wv; x = num / np.maximum(den, 1e-9)
        cl = ((p['num'] - x[:, None] * p['den']) ** 2) @ wv                       # match-clustered variance of the sum, centred on the PLAYER'S OWN mean
        out[k] = (num, den, np.maximum(cl, p['sig2'] * (p['iid'] @ wv)))          # never below the ball-level (independent) variance
    return out
def xv(t):
    num, den, vnum = t; D = np.maximum(den, 1e-9); return np.where(den > 1e-9, num / D, np.nan), vnum / D ** 2          # match-clustered variance (balls within an innings are not independent)
def estimates(T):
    E = {}
    if BAT:
        E['rae'] = xv(T['rae']); E['dc'] = xv(T['dc'])
        for k in ('wpa_cc', 'big', 'relief', 'hit', 'floor', 'finish', 'protect'): E[k] = xv(T[k])
        xp, vp = xv(T['imp_p']); xn, vn = xv(T['imp_n']); E['delta'] = (xp - xn, vp + vn)
    else:
        E['runs'] = xv(T['runs']); E['wdc'] = xv(T['wdc'])
        for k in ('win_cl', 'qwk', 'creation', 'bounce', 'closing', 'hit', 'floor'): E[k] = xv(T[k])
        xp, vp = xv(T['runs']); xn, vn = xv(T['rn']); E['delta'] = (xp - xn, vp + vn)
    return E
def eb(x, v, ok, fixed=None):
    good = ok & np.isfinite(x) & np.isfinite(v) & (v > 0); xs, vs = x[good], v[good]
    if fixed is None:
        wi = 1 / vs; mu0 = (wi * xs).sum() / wi.sum(); Q = (wi * (xs - mu0) ** 2).sum()
        tau2 = max(0.0, (Q - (len(xs) - 1)) / (wi.sum() - (wi ** 2).sum() / wi.sum())); mu = ((1 / (tau2 + vs)) * xs).sum() / (1 / (tau2 + vs)).sum()
    else: tau2, mu = fixed
    pm_ = np.zeros(len(x)); ps = np.full(len(x), np.nan); B = np.zeros(len(x))
    B[good] = tau2 / (tau2 + v[good]); pm_[good] = B[good] * (x[good] - mu); ps[good] = np.sqrt(B[good] * v[good]); return pm_, ps, np.sqrt(tau2), B, (tau2, mu)
SUMS = {'imp': ('rae', 'dc')} if BAT else {'runs': ('runs', 'wdc')}          # metrics made of two separately shrunk pieces (scoring + cost of dismissal / runs saved + wicket value)
RAW = ['rae', 'dc'] if BAT else ['runs', 'wdc']
def components(T, ok, fixed=None, rec=None):
    E = estimates(T); Z = {}; S = {}; PMd = {}; TAU = {}; BB = {}
    for k in E:
        pm_, ps, t, B, hp = eb(*E[k], ok, None if fixed is None else fixed[k]); PMd[k] = (pm_, ps); BB[k] = B
        if rec is not None: rec[k] = hp
        if k in RAW and k not in WN: continue                       # raw pieces of a SUM metric: not scored on their own
        TAU[k] = t; Z[k] = pm_ / t if t > 1e-12 else np.zeros(NP); S[k] = ps / t if t > 1e-12 else np.full(NP, np.nan)
    for k, mem in SUMS.items():
        pm_i = sum(PMd[x][0] for x in mem); sd_i = np.sqrt(sum(np.nan_to_num(PMd[x][1], nan=0.0) ** 2 for x in mem)); good = ok & (BB[mem[0]] > 0)
        t = fixed['_sum_' + k] if fixed is not None else (pm_i[good].std() if good.sum() > 5 else 0.0)
        if rec is not None: rec['_sum_' + k] = t
        TAU[k] = t; Z[k] = pm_i / t if t > 1e-12 else np.zeros(NP); S[k] = np.where(good, sd_i / t, np.nan) if t > 1e-12 else np.full(NP, np.nan)
    if BAT:                                                       # strike protection: no credit if the set batter scored below expectation on the balls he took
        gm = np.where(T['gR'][1] > 0, T['gR'][0] / np.maximum(T['gR'][1], 1e-9), 0.0); Z['protect'] = np.where(gm < 0, np.minimum(Z['protect'], 0), Z['protect'])
    return np.array([Z[k] for k in CK]), np.array([S[k] for k in CK]), np.array([TAU[k] for k in CK]), E
def spear(a, b, ok):
    ix = np.where(ok)[0]
    if len(ix) < 5 or np.std(a[ix]) < 1e-12 or np.std(b[ix]) < 1e-12: return 0.0
    return float(spearmanr(a[ix], b[ix])[0])

# ------------------------------------------------------------------ base build
w_base = row_w()
P0 = build(PRESS, w_base); ones = np.ones(NM); T_full = totals(P0, ones)
nP, nInn = T_full['n'][0], T_full['inn'][1]
NP_MIN, PINN_MIN = (60, 8) if BAT else (80, 8)
ok_pop = nP >= 25
grid = {(a, b): int(((nP >= a) & (nInn >= b)).sum()) for a in (40, 60, 80, 100, 150) for b in (6, 8, 10, 15)}
print('players with strict-pressure sample (balls>=a, innings>=b):', grid)
Z0, S0, TAU0, E0 = components(T_full, ok_pop); REC0 = {}; components(T_full, ok_pop, rec=REC0)
PB = build(BROAD, w_base); T_fullB = totals(PB, ones); ok_popB = T_fullB['n'][0] >= 25; ZB0, _, TAUB, _ = components(T_fullB, ok_popB); RECB = {}; components(T_fullB, ok_popB, rec=RECB)
if TIER != 'strict': raise SystemExit('v4 headline is the strict/broad ensemble; run with TIER=strict (default)')
print('tau (population signal sd):', dict(zip(CK, np.round(TAU0, 4))))
elig = (nP >= NP_MIN) & (nInn >= PINN_MIN); eidx = np.where(elig)[0]
lowconf = (nP < (100 if BAT else 120)) | (nInn < 10)
print('eligible', len(eidx), '| low-confidence among eligible', int(lowconf[eidx].sum()))

# ------------------------------------------------------------------ slots / roles
if BAT:
    fac = r.groupby('pl').agg(balls=('runs', 'size'), ord_med=('batting_order', 'median'))
    def slot_of(o): return 0 if o < 3.5 else (1 if o <= 5 else 2)
    GRP = ['Top order (1–3)', 'Middle order (4–5)', 'Finisher (6–11)']; grp_p = np.array([slot_of(fac.ord_med[p]) for p in players])
else:
    lg = r[r.legal == 1]; dsh = lg.groupby('pl').over.apply(lambda x: (x >= 15).mean()).reindex(players).fillna(0).values; psh = lg.groupby('pl').over.apply(lambda x: (x <= 5).mean()).reindex(players).fillna(0).values
    GRP = ['Powerplay bowler (overs 1–6)', 'Middle-overs bowler (7–15)', 'Death bowler (16–20)']; grp_p = np.where(dsh >= 0.30, 2, np.where(psh >= 0.45, 0, 1))

# ------------------------------------------------------------------ validation: split-half repeatability, held-out validity, chronological
def target_Y(T, ok):
    E = estimates(T)
    def rz(x): o = np.full(NP, np.nan); g = ok & np.isfinite(x); o[g] = np.clip((x[g] - x[g].mean()) / (x[g].std() + 1e-12), -3, 3); return np.nan_to_num(o)
    if BAT: return 0.5 * (rz(E['rae'][0] + E['dc'][0]) + rz(xv(T['wpa_cc'])[0]))
    return 0.5 * (rz(E['runs'][0] + E['wdc'][0]) + rz(xv(T['win_cl'])[0]))
def comp_score(Z, W=W0): return (W[:, None] * Z).sum(0)
lv = np.array([1.0 if BLOCK[k] == 'Level' else 0 for k in CK]); nd = np.array([0.0 if k == 'delta' else WN[k] for k in CK])
dtsm = np.array([str(dts[x])[:4] >= '2024' for x in mids]); seeds = list(range(300, 320))
def validate(P, okmin=25):
    rep = np.zeros(len(CK)); val = np.zeros(len(CK)); R_ = {k: [] for k in ('composite (final weights)', 'equal weights over the 9 metrics', 'Level metrics only', 'final weights without Delta', 'composite split-half repeatability')}
    for s_ in seeds:
        sel = (np.random.default_rng(s_).random(NM) < .5).astype(float); Ta = totals(P, sel); Tb = totals(P, 1 - sel); ok_ = (Ta['n'][0] >= okmin) & (Tb['n'][0] >= okmin)
        Za, _, _, _ = components(Ta, ok_); Zb, _, _, _ = components(Tb, ok_); Y = target_Y(Tb, ok_)
        for j in range(len(CK)): rep[j] += spear(Za[j], Zb[j], ok_) / len(seeds); val[j] += spear(Za[j], Y, ok_) / len(seeds)
        R_['composite (final weights)'].append(spear(comp_score(Za), Y, ok_)); R_['equal weights over the 9 metrics'].append(spear(comp_score(Za, np.ones(len(CK)) / len(CK)), Y, ok_)); R_['Level metrics only'].append(spear(comp_score(Za, lv / lv.sum()), Y, ok_))
        R_['final weights without Delta'].append(spear(comp_score(Za, nd / nd.sum()), Y, ok_)); R_['composite split-half repeatability'].append(spear(comp_score(Za), comp_score(Zb), ok_))
    er = {}
    for lab, (fm, tm_) in {'2022-23 → 2024-26': (~dtsm, dtsm), '2024-26 → 2022-23': (dtsm, ~dtsm)}.items():
        Ta = totals(P, fm.astype(float)); Tb = totals(P, tm_.astype(float)); ok_ = (Ta['n'][0] >= okmin) & (Tb['n'][0] >= okmin); Za, _, _, _ = components(Ta, ok_); Y = target_Y(Tb, ok_)
        er[lab] = {'composite': round(spear(comp_score(Za), Y, ok_), 3), 'Level only': round(spear(comp_score(Za, lv / lv.sum()), Y, ok_), 3), 'n': int(ok_.sum())}
    return rep, val, {k: round(float(np.mean(v)), 3) for k, v in R_.items()}, er
rep, val, pred, era = validate(P0); repB, valB, predB, eraB = validate(PB)
print('STRICT component repeat  ', dict(zip(CK, np.round(rep, 2)))); print('STRICT component validity', dict(zip(CK, np.round(val, 2)))); print('STRICT composite', pred, era)
print('BROAD  component repeat  ', dict(zip(CK, np.round(repB, 2)))); print('BROAD  component validity', dict(zip(CK, np.round(valB, 2)))); print('BROAD  composite', predB, eraB)

# ------------------------------------------------------------------ final composite + stress test: match bootstrap x +-35% weight perturbation x pressure-tier (strict / broad) drawn at random; shrinkage hyper-parameters held fixed
def zs_(c): x = c[eidx]; return (x - x.mean()) / (x.std() + 1e-9)
zS, zBr = zs_(comp_score(Z0)), zs_(comp_score(ZB0)); base_z = 0.5 * (zS + zBr); comp0 = comp_score(Z0); mu_c, sd_c = comp0[eidx].mean(), comp0[eidx].std()
Z0M = 0.5 * (Z0 + ZB0)                                                         # per-metric Z averaged over the two tiers (for the block breakdown shown per player)
rng = np.random.default_rng(7); BM = np.stack([np.bincount(rng.integers(0, NM, NM), minlength=NM).astype(float) for _ in range(BR)]); WP = np.exp(rng.normal(0, SIGW, (BR, len(CK)))) * W0; WP = WP / WP.sum(1, keepdims=True)
def stress(eidx_):
    zb = np.zeros((BR, len(eidx_)))
    for i in range(BR):
        P, rc, okp = (P0, REC0, ok_pop) if i % 2 == 0 else (PB, RECB, ok_popB)
        Tb = totals(P, BM[i]); Zb, _, _, _ = components(Tb, okp, fixed=rc); c = comp_score(Zb, WP[i]); zb[i] = (c[eidx_] - c[eidx_].mean()) / (c[eidx_].std() + 1e-9)
    return zb
ZB = stress(eidx)
ranks = np.zeros(ZB.shape, int)
for i in range(len(ZB)): ranks[i][np.argsort(-ZB[i])] = np.arange(1, ZB.shape[1] + 1)
top10p = (ranks <= 10).mean(0); med_rank = np.median(ranks, axis=0); rlo = np.percentile(ranks, 5, axis=0); rhi = np.percentile(ranks, 95, axis=0); mean_z = ZB.mean(0)
# final rule: membership by P(top-10), order by median rank (ties: P(top-10), then base composite)
# FINAL RULE (v4.1): rank strictly by the Clutch Meter (= 50 + 10 x mean standardised score over the stress runs). P(top-10) and the rank range are shown as confidence, not used to order.
order_cm = np.argsort(-mean_z, kind='stable'); final_order = list(order_cm[:10]); final_pos = {int(i): k + 1 for k, i in enumerate(order_cm)}
print('stress: mean P(top10) of final 10 =', round(float(top10p[final_order].mean()), 3), '| players with P(top10)>=0.5:', int((top10p >= 0.5).sum()), '| >=0.8:', int((top10p >= 0.8).sum()))

# ------------------------------------------------------------------ sensitivity variants (overlap with final top 10 / top 30), each re-estimated
final10 = set(eidx[final_order]); order_base = np.argsort(-base_z); fin30 = set(eidx[[i for i, _ in sorted(final_pos.items(), key=lambda t: t[1])][:30]])
sens = {}
def variant(name, press=PRESS, w=w_base, Wv=W0, colmask=None, role_adj=False):
    P = P0 if (press is PRESS and w is w_base) else build(press, w); cm = np.ones(NM) if colmask is None else colmask.astype(float)
    T = totals(P, cm); ok_ = T['n'][0] >= (25 if colmask is None else 15); Z, _, _, _ = components(T, ok_)
    if colmask is None: ee = eidx
    else: ee = np.where((T['n'][0] >= NP_MIN * 0.6) & (T['inn'][1] >= max(4, PINN_MIN * 0.6)))[0]
    c = comp_score(Z, Wv).copy()
    if role_adj:
        for g in range(len(GRP)):
            ix = [i for i in ee if grp_p[i] == g]
            if len(ix) > 2: c[ix] = c[ix] - c[ix].mean()
    order = ee[np.argsort(-c[ee])]; t10 = list(order[:10]); sens[name] = {'o10': len(final10 & set(t10)), 'o30': len(fin30 & set(order[:30])), 'n': int(len(ee)), 'top10': [str(nm_[players[i]]) for i in t10]}
nm_ = r.groupby('pl').name.first()
variant('Role-standardised (score centred within batting slot / bowling role)', role_adj=True); variant('Strict tier only (~17% of balls)'); variant('Broad tier only (v3 definition, ~33% of balls)', press=BROAD)
variant('No match-importance weighting', w=row_w(use_imp=False)); variant('No recency weighting', w=row_w(use_rec=False))
variant('Equal weights over the 9 metrics', Wv=np.ones(len(CK)) / len(CK))
lv_ = np.array([1.0 if BLOCK[k] == 'Level' else 0 for k in CK]); variant('Level metrics only (no Delta / Reliability / extras)', Wv=lv_ / lv_.sum())
nd_ = np.array([0 if k == 'delta' else WN[k] for k in CK], float); variant('Without Delta', Wv=nd_ / nd_.sum())
rl_ = np.array([0 if BLOCK[k] in ('Relief', 'Tail') else WN[k] for k in CK], float); variant('Without Relief and Tail extras', Wv=rl_ / rl_.sum())
ipl_m = np.array([bool(r[r.match_id == x].is_ipl.iloc[0]) for x in mids]) if False else (r.groupby('match_id').is_ipl.first().reindex(mids).values > 0.5)
variant('IPL matches only', colmask=ipl_m); variant('T20I / other matches only', colmask=~ipl_m)
print('sensitivity overlap with final top 10:', {k: (v['o10'], v['n']) for k, v in sens.items()})

# ------------------------------------------------------------------ role-bias audit
audit = {}
for g in range(len(GRP)):
    ix = [i for i in eidx if grp_p[i] == g]
    if ix: audit[GRP[g]] = {'eligible': len(ix), 'mean_z': round(float(base_z[[list(eidx).index(i) for i in ix]].mean()), 2), 'in_top10': int(sum(1 for i in final_order if grp_p[eidx[i]] == g)), 'in_top30': int(sum(1 for p_, i in final_pos.items() if i <= 30 and grp_p[eidx[p_]] == g))}
print('ROLE AUDIT', audit)
# the same audit on the v3 definition of the score, for contrast is not recomputed here; dashboard reports v4 only

# ------------------------------------------------------------------ player records
def stats_bat(df):
    n = len(df)
    if n == 0: return None
    return {'n': int(n), 'sr': round(100 * df.runs.mean(), 1), 'esr': round(100 * df.e_runs.mean(), 1), 'dot': round(100 * df.a_dot.mean(), 1), 'bnd': round(100 * df.a_bnd.mean(), 1), 'outs': int(df.a_out.sum()), 'rae': round(100 * df.rae.mean(), 1), 'iar': round(100 * df.iar.mean(), 1), 'waa': round(float(df.waa.sum()), 2)}
def stats_bowl(df):
    lb = float(df.legal.sum())
    if lb == 0: return None
    ch = float(df.ch.sum()); return {'balls': int(lb), 'runs': int(ch), 'wkts': int(df.cw.sum()), 'econ': round(6 * ch / lb, 2), 'eecon': round(6 * float(df.ech.sum()) / lb, 2), 'dot': round(100 * float(df.dotb.sum()) / lb, 1), 'edot': round(100 * float(df.pd.sum()) / lb, 1), 'bnd': round(100 * float(df.bnd.sum()) / lb, 1), 'ebnd': round(100 * float(df.pbnd.sum()) / lb, 1), 'ewk': round(float(df.pw.sum()), 1), 'win': round(float((df.bwpa - df.ebw).sum()), 2)}
ST = stats_bat if BAT else stats_bowl
ipl26 = set(pd.read_csv(f'{DATA}/competition_team_player_mapping.csv', keep_default_na=False, na_values=[]).query("comp_name == 'IPL 2026'").player_id)
pe = pd.read_csv(f'{DATA}/playing_eleven.csv', keep_default_na=False, na_values=[]); pe = pe[pe.is_played.astype(str).str.lower() == 'true'].merge(m[['match_id', 'match_date', 'match_format']], on='match_id')
last_played = pe.groupby('player_id').match_date.max(); intl26 = set(pe[(pe.match_format == 'T20I') & (pe.match_date >= '2026-01-01')].player_id)
tm = r.groupby(['pl', 'bat_team' if BAT else 'bowling_team_id']).size().reset_index(name='n').sort_values(['pl', 'n'], ascending=[True, False]); tcol = 'bat_team' if BAT else 'bowling_team_id'
G = r.groupby('pl'); posall = {int(players[i]): k + 1 for k, i in enumerate(np.argsort(-comp0))}
eligpos = {int(eidx[i]): i for i in range(len(eidx))}
BLK = ['Level', 'Relief', 'Delta', 'Reliability', 'Tail']; Bmat = np.array([[1.0 if BLOCK[k] == b else 0 for k in CK] for b in BLK])
Zcont = Z0M * W0[:, None]                                              # weighted contribution of each metric to each player's composite
PT_ = r.groupby('pl').pace_or_spin.agg(lambda x: x.value_counts().index[0]) if not BAT else None; BT_ = r.groupby('pl').bowling_type.agg(lambda x: x.value_counts().index[0]) if not BAT else None
recs = []
for p in players:
    i = pidx[p]; df = G.get_group(p); nb = int(df.legal.sum()) if not BAT else len(df)
    if nb < 60: continue
    pr = df[df.press == 1]; nr = df[df.press == 0]; isel = i in eligpos
    rec = {'id': int(p), 'name': str(nm_[p]), 'teams': [tname.get(t, str(t)) for t in tm[tm.pl == p][tcol].head(2)], 'grp': GRP[grp_p[i]], 'balls': nb, 'elig': bool(isel), 'active': bool((int(p) in ipl26) or (int(p) in intl26)), 'last': str(last_played.get(int(p), ''))[:10],
           'pn': int(nP[i]), 'pinn': int(nInn[i]), 'press': ST(pr), 'norm': ST(nr), 'ipl': round(float(df.is_ipl.mean()), 3), 'rank_all': posall.get(int(p)), 'lowconf': bool(lowconf[i]),
           'blocks': [round(float(x), 2) for x in Bmat @ Zcont[:, i]], 'z': [round(float(x), 2) for x in Z0M[:, i]]}
    if not BAT: rec['ptype'] = str(PT_[p]); rec['btype'] = str(BT_[p])
    if BAT: rec['types'] = {d['TYPES'][t]: stats_bat(pr[pr.ptype == t]) for t in range(len(d['TYPES'])) if (pr.ptype == t).sum() >= 20}
    if isel:
        k = eligpos[i]; rec.update({'rank': final_pos[k], 'top10p': round(float(top10p[k]), 3), 'medrank': float(med_rank[k]), 'rlo': int(rlo[k]), 'rhi': int(rhi[k]), 'score': round(50 + 10 * float(mean_z[k]), 1), 'score_base': round(50 + 10 * float(base_z[k]), 1),
                                    'sd': round(float(ZB[:, k].std() * 10), 2)})
    recs.append(rec)
# ------------------------------------------------------------------ league-wide: pressure vs ordinary (the brief asks to SHOW the tiers differ)
def tier_tab(df):
    out = {}
    for lab, d_ in (('Pressure (strict tier)', df[df.press == 1]), ('Ordinary', df[df.press == 0]), ('Broad tier (v3)', df[df.press_broad == 1])):
        if BAT: out[lab] = {'share': round(len(d_) / len(df), 3), 'strike rate': round(100 * d_.runs.mean(), 1), 'dismissal % per ball': round(100 * d_.a_out.mean(), 2), 'dot %': round(100 * d_.a_dot.mean(), 1), 'boundary %': round(100 * d_.a_bnd.mean(), 1), 'sd of win-prob swing per ball (x100)': round(100 * d_.wpa.std(), 2), 'batting-side win %': round(100 * d_.bat_won.mean(), 1)}
        else:
            lb = d_.legal.sum(); out[lab] = {'share': round(len(d_) / len(df), 3), 'economy': round(6 * d_.ch.sum() / lb, 2), 'wickets per over': round(6 * d_.cw.sum() / lb, 3), 'dot %': round(100 * d_.dotb.sum() / lb, 1), 'boundary %': round(100 * d_.bnd.sum() / lb, 1), 'wides+no-balls %': round(100 * d_.xt.mean(), 2), 'fielding-side win %': round(100 * (1 - d_.bat_won.mean()), 1)}
    return out
tiers = tier_tab(r)
tiers_by_inn = {int(i): tier_tab(g_) for i, g_ in r.groupby('innings')}
by_over = [[int(o), round(float(g_.press.mean()), 3)] for o, g_ in r.assign(o=(r.lb // 6)).groupby('o')]

# ------------------------------------------------------------------ VIEWS: every (from-year, to-year, league) window is RE-SCORED on its own matches (shrinkage hyper-parameters fixed from the full data, so small windows are shrunk, not zeroed)
yrs = np.array([int(str(dts[x])[:4]) for x in mids])
if BAT: S_b = mat(PRESS * 1.0); S_r = mat(PRESS * r.runs.values); S_e = mat(PRESS * r.e_runs.values)
else: S_b = mat(PRESS * L); S_r = mat(PRESS * r.ch.values); S_e = mat(PRESS * r.ech.values)
BRV = int(os.environ.get('BRV', 60)); rec_ids = {x['id'] for x in recs}; views = {}
def zs_e(c, ev): x = c[ev]; return (x - x.mean()) / (x.std() + 1e-9)
def order_rule(mz_): return {int(i): k + 1 for k, i in enumerate(np.argsort(-mz_, kind='stable'))}
for y0 in range(2022, 2027):
    for y1 in range(y0, 2027):
        for lg in ('all', 'ipl', 'intl'):
            key = f'{y0}-{y1}|{lg}'; cm = ((yrs >= y0) & (yrs <= y1) & (ipl_m if lg == 'ipl' else (~ipl_m if lg == 'intl' else True))).astype(float)
            T0v = totals(P0, cm); TBv = totals(PB, cm); nPv = T0v['n'][0]; nIv = T0v['inn'][1]; frac = nPv.sum() / nP.sum()
            minb = max(25.0, NP_MIN * frac); mini = max(3, int(round(PINN_MIN * frac))); ev = np.where((nPv >= minb) & (nIv >= mini))[0]
            if len(ev) < 8: views[key] = {'n': int(len(ev)), 'rows': []}; continue
            okv = nPv >= 10; okb = TBv['n'][0] >= 10
            Zv, _, _, _ = components(T0v, okv, fixed=REC0); ZBv, _, _, _ = components(TBv, okb, fixed=RECB)
            base = 0.5 * (zs_e(comp_score(Zv), ev) + zs_e(comp_score(ZBv), ev))
            if (y0, y1, lg) == (2022, 2026, 'all'): ev = eidx; t10, md, lo_, hi_, mz = top10p, med_rank, rlo, rhi, mean_z; pos = {k: final_pos[k] for k in range(len(ev))}
            else:
                zb = np.zeros((BRV, len(ev)))
                for i in range(BRV):
                    Pq, rc, okq = (P0, REC0, okv) if i % 2 == 0 else (PB, RECB, okb); Tq = totals(Pq, cm * BM[i]); Zq, _, _, _ = components(Tq, okq, fixed=rc); c = comp_score(Zq, WP[i]); zb[i] = zs_e(c, ev)
                rk = np.zeros(zb.shape, int)
                for i in range(BRV): rk[i][np.argsort(-zb[i])] = np.arange(1, len(ev) + 1)
                t10 = (rk <= 10).mean(0); md = np.median(rk, axis=0); lo_ = np.percentile(rk, 5, axis=0); hi_ = np.percentile(rk, 95, axis=0); mz = zb.mean(0); pos = order_rule(mz)
            hd = np.where(T0v['hit'][1] > 0, T0v['hit'][0] / np.maximum(T0v['hit'][1], 1e-9), np.nan); rows = []
            for k, i in enumerate(ev):
                pid = int(players[i])
                if pid not in rec_ids: continue
                bl = np.maximum(S_b[i] @ cm, 1e-9); rn = S_r[i] @ cm; ex = S_e[i] @ cm
                a = 100 * rn / bl if BAT else 6 * rn / bl; im = 100 * (rn - ex) / bl if BAT else 6 * (ex - rn) / bl
                rows.append([pid, int(pos[k]), round(50 + 10 * float(mz[k]), 1), round(float(t10[k]), 3), int(lo_[k]), int(hi_[k]), int(nPv[i]), int(nIv[i]), int(nPv[i] < (100 if BAT else 120) or nIv[i] < 10), round(float(a), 2 if not BAT else 1), round(float(im), 2), None if np.isnan(hd[i]) else round(float(hd[i]), 2)])
            views[key] = {'n': int(len(ev)), 'minb': round(minb), 'mini': mini, 'rows': sorted(rows, key=lambda x: x[1])}
print('views built:', len(views), '| rows in 2026 / all:', len(views['2026-2026|all']['rows']), len(views['2022-2026|intl']['rows']))
cz = comp_score(Z0M); eff = [float(np.cov(W0[j] * Z0M[j][eidx], cz[eidx])[0, 1] / (cz[eidx].var() + 1e-12)) for j in range(len(CK))]
meta = {'views': views, 'effShare': [round(x, 3) for x in eff], 'kind': KIND, 'matches': int(NM), 'rows': int(len(r)), 'players': int(NP), 'eligible': int(len(eidx)), 'lowconf': int(lowconf[eidx].sum()), 'tier': TIER, 'strictShare': round(float(STRICT.mean()), 3), 'broadShare': round(float(BROAD.mean()), 3), 'minBalls': NP_MIN, 'minInnings': PINN_MIN,
        'groups': GRP, 'groupCounts': {GRP[g]: int((grp_p[eidx] == g).sum()) for g in range(len(GRP))}, 'metrics': {'keys': CK, 'names': [CN[k] for k in CK], 'weights': [float(x) for x in W0], 'block': [BLOCK[k] for k in CK], 'tau': [round(float(x), 5) for x in TAU0],
        'repeat': [round(float(x), 2) for x in rep], 'validity': [round(float(x), 2) for x in val], 'tau_broad': [round(float(x), 5) for x in TAUB], 'repeat_broad': [round(float(x), 2) for x in repB], 'validity_broad': [round(float(x), 2) for x in valB]}, 'blocks': BLK, 'pred': pred, 'predBroad': predB, 'era': era, 'eraBroad': eraB, 'sens': sens, 'audit': audit, 'tiers': tiers, 'tiersByInnings': tiers_by_inn, 'pressureByOver': by_over,
        'eligGrid': {f'{a}/{b}': v for (a, b), v in grid.items()}, 'weightSigma': SIGW, 'bootstrap': BR, 'top10pFinal': [round(float(top10p[i]), 3) for i in final_order],
        'wp1': d.get('wp1'), 'wp2': d.get('wp2'), 'cal': d.get('cal'), 'dq': d.get('dq'), 'dataEnd': str(m.match_date.max()), 'types': d.get('TYPES') if BAT else d.get('BT')}
json.dump({'meta': meta, 'players': recs}, open(OUT, 'w'), separators=(',', ':'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))
tops = sorted([x for x in recs if x['elig']], key=lambda x: x['rank'])
print('\nFINAL', KIND)
for x in tops[:20]: print(x['rank'], x['name'], '|', x['grp'][:9], '| score', x['score'], '| P10', x['top10p'], '| med', x['medrank'], '[%d-%d]' % (x['rlo'], x['rhi']), '| pn', x['pn'], 'pinn', x['pinn'], '| blocks', x['blocks'], 'LOWCONF' if x['lowconf'] else '')
print('groups in top 10:', {g: sum(1 for x in tops[:10] if x['grp'] == g) for g in GRP})
