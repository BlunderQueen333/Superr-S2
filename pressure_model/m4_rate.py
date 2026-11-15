"""Stage 3 (v3), batters. A player's pressure impact = USUAL LEVEL (all calm balls) + SHRUNK PRESSURE DEVIATION, in runs, counting the cost of dismissal.

Per-ball residuals vs the expected outcome of that slot / partner / bowler / state: rae (runs above expected), dc (dismissal cost = change in expected future runs),
oae, waa. Every estimate has a known sampling variance; empirical-Bayes (DerSimonian-Laird) shrinkage pools each toward the population by exactly its noise, so a component
with no real between-player spread is shrunk to zero. Scoring and dismissal cost are shrunk SEPARATELY and added in runs (imph).
Scored pillars: impact (70%) and consistency (30%); slope, survival and strike protection are tested and reported but not scored (no persistent signal).
Published rank = average of 5 ensemble variants (base, pressure top 30% / 15%, no match importance, pressure-tilted). Rank range and top-10 chance = bootstrap over matches of the
whole ensemble with the shrinkage hyper-parameters held FIXED (re-estimating them on duplicated matches inflates the apparent spread of players).
ACTIVE_ONLY=1 restricts the ranked pool to IPL 2026 squad members / 2026 T20I players (default: everyone).
"""
import sys, pickle, json, os
PYLIB, IN, DATA, OUT = sys.argv[1:5]
if PYLIB != '-': sys.path.insert(0, PYLIB)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
d = pickle.load(open(IN, 'rb')); bb = d['bb'].reset_index(drop=True); e = d['e']; m = d['m']; TYPES = d['TYPES']; sp = d['sp']; ord_med = d['ord_med']
HERE = os.path.dirname(os.path.abspath(__file__))
teams = pd.read_csv(f'{DATA}/teams.csv'); tname = dict(zip(teams.team_id, teams.team_name))
import official_strength as OS
_, _, tn_, _, _, s_ipl, s_icc = OS.build(m, teams, HERE)
_bt = pd.to_numeric(bb.bowling_team_id, errors='coerce').values
bb['strong'] = np.array([(s_ipl.get((c_, t_), 99) <= 4) if i_ == 1 else (s_icc.get((int(y_), tn_.get(t_)), 99) <= 4) for c_, t_, i_, y_ in zip(bb.comp_id, _bt, bb.is_ipl, bb.year)])
players = sorted(bb.bat.unique()); pidx = {p: i for i, p in enumerate(players)}; midx = {x: i for i, x in enumerate(sorted(bb.match_id.unique()))}
NP, NM = len(players), len(midx)
pi_i = bb.bat.map(pidx).astype(int).values; mi_ = bb.match_id.map(midx).astype(int).values
w_imp = (0.5 + bb.imp.values); w_imp = w_imp / w_imp.mean(); w_one = np.ones(len(bb))
bb['ig'] = bb.groupby(['match_id', 'innings', 'bat'], sort=False).ngroup(); ig = bb.ig.values; NI = int(ig.max()) + 1
# ---- active players
mp = pd.read_csv(f'{DATA}/competition_team_player_mapping.csv', keep_default_na=False, na_values=[]); ipl26 = set(mp[mp.comp_name == 'IPL 2026'].player_id)
pe = pd.read_csv(f'{DATA}/playing_eleven.csv', keep_default_na=False, na_values=[]); pe = pe[pe.is_played.astype(str).str.lower() == 'true']
pe = pe.merge(m[['match_id', 'match_date', 'match_format']], on='match_id'); last_played = pe.groupby('player_id').match_date.max()
intl26 = set(pe[(pe.match_format == 'T20I') & (pe.match_date >= '2026-01-01')].player_id)
active_p = np.array([(p in ipl26) or (p in intl26) for p in players])
if os.environ.get('ACTIVE_ONLY', '0') != '1': active_p = np.ones(len(active_p), bool)          # default: EVERY player 2022-2026; ACTIVE_ONLY=1 restricts to IPL 2026 squads / 2026 T20I players
# ---- slot (board) of each player: usual batting position
def slot_of(o): return 0 if o <= 2 else (1 if o < 3.5 else (2 if o <= 5 else (3 if o <= 7 else 4)))
SN = ['Opener (1–2)', 'Number 3', 'Middle order (4–5)', 'Finisher (6–7)', 'Lower order (8–11)']
fac = bb.groupby('bat').agg(balls=('runs', 'size'), ord_med=('batting_order', 'median'))
slot_p = np.array([slot_of(fac.ord_med[p]) for p in players])
# ---- per-ball residuals
R = {'rae': bb.rae.values, 'dc': (bb.iar - bb.rae).values, 'oae': bb.oae.values, 'waa': bb.waa.values, 'iar': bb.iar.values}      # dc = cost of dismissal (change in expected future runs), shrunk SEPARATELY from scoring
X = bb.pix.values - 0.5; INN2 = bb.innings.values == 2; STRONG = bb.strong.values
def mat(v, rows=pi_i, cols=mi_):
    A = np.zeros((NP, NM)); np.add.at(A, (rows, cols), v); return A
# ---- protect: strike-protection residuals
sp = sp.copy(); sp['pi'] = sp.setb.map(pidx); sp['mi'] = sp.match_id.map(midx); sp = sp.dropna(subset=['pi', 'mi'])
sp['res'] = sp.faced - sp.groupby('over').faced.transform('mean'); SPR = sp.res.values; SPI = sp.pi.astype(int).values; SPM = sp.mi.astype(int).values
def build(press, w):
    """all additive per-(player, match) matrices for a given pressure mask and match weights"""
    pm = press.astype(bool); M = {}
    wp = w * pm; M['W'] = mat(wp); M['W2'] = mat(w * w * pm); M['n'] = mat(pm.astype(float))
    for k, r in R.items(): M['P' + k] = mat(wp * r)
    for k in ('waa', 'iar'):
        M['aW' + k] = mat(w); M['aX' + k] = mat(w * X); M['aXX' + k] = mat(w * X * X); M['aR' + k] = mat(w * R[k]); M['aXR' + k] = mat(w * X * R[k])
    nm_ = ~pm; M['nW'] = mat(w * nm_); M['nW2'] = mat(w * w * nm_)              # NORMAL (non-pressure) balls: the player's usual level
    for k_ in ('rae', 'dc'): M['N' + k_] = mat(w * nm_ * R[k_])
    M['uN'] = mat(np.ones(len(bb))); M['uX'] = mat(X); M['uXX'] = mat(X * X)
    for tag, mk in (('s', STRONG & pm), ('c', INN2 & pm)):
        M[tag + 'W'] = mat(w * mk); M[tag + 'W2'] = mat(w * w * mk); M[tag + 'R'] = mat(w * mk * R['rae'])
    # innings units: pressure innings = >= 6 pressure balls; hit = non-negative win impact
    pb = np.bincount(ig, weights=pm.astype(float), minlength=NI); ws = np.bincount(ig, weights=R['waa'] * pm, minlength=NI)
    first = pd.Series(np.arange(len(bb))).groupby(ig).first().values
    inn = (pb >= 6).astype(float); hit = inn * (ws >= 0); flo = inn * np.minimum(ws, 0)
    M['inn'] = mat(inn[ig[first]], pi_i[first], mi_[first]); M['hit'] = mat(hit[ig[first]], pi_i[first], mi_[first]); M['flo'] = mat(flo[ig[first]], pi_i[first], mi_[first])
    M['pN'] = np.zeros((NP, NM)); np.add.at(M['pN'], (SPI, SPM), 1.0); M['pR'] = np.zeros((NP, NM)); np.add.at(M['pR'], (SPI, SPM), SPR)
    sig = {}
    for k, r in R.items():
        mu = (wp * r).sum() / wp.sum(); sig[k] = float((wp * (r - mu) ** 2).sum() / wp.sum())
    for k_ in ('rae', 'dc'): wn = w * nm_; mu_ = (wn * R[k_]).sum() / wn.sum(); sig[k_ + '_n'] = float((wn * (R[k_] - mu_) ** 2).sum() / wn.sum())
    sig['slope_waa'] = float(R['waa'].var()); sig['slope_iar'] = float(R['iar'].var())
    pv = ws[inn > 0]; sig['hit_p'] = float((pv >= 0).mean()); fv = np.minimum(pv, 0); sig['floor'] = float(fv.var()); sig['protect'] = float(SPR.var())
    return M, sig
TAUD = 0.0          # floor on the between-player spread of the pressure deviation (0 = let the data decide; >0 = pressure-tilted sensitivity variant)
def eb(x, v, ok, tmin=0.0, fixed=None):
    """DerSimonian-Laird empirical Bayes. fixed=(tau2, mu): reuse hyper-parameters (bootstrap: duplicates of matches inflate the apparent spread of players, so tau must not be re-estimated)"""
    good = ok & np.isfinite(x) & np.isfinite(v) & (v > 0)
    xs, vs = x[good], v[good]
    if fixed is None:
        wi = 1 / vs; mu0 = (wi * xs).sum() / wi.sum(); Q = (wi * (xs - mu0) ** 2).sum()
        tau2 = max(tmin ** 2, (Q - (len(xs) - 1)) / (wi.sum() - (wi ** 2).sum() / wi.sum())); mu = ((1 / (tau2 + vs)) * xs).sum() / (1 / (tau2 + vs)).sum()
    else: tau2, mu = fixed
    pm_ = np.zeros(len(x)); ps = np.full(len(x), np.nan); Bi = np.zeros(len(x))
    Bi[good] = tau2 / (tau2 + v[good]); pm_[good] = Bi[good] * (x[good] - mu); ps[good] = np.sqrt(Bi[good] * v[good])
    return pm_, ps, np.sqrt(tau2), Bi, (tau2, mu)
CK = ['rae', 'dc', 'imp', 'rae_n', 'dc_n', 'rae_d', 'dc_d', 'imph', 'oae', 'waa', 'slope_waa', 'slope_iar', 'hit', 'floor', 'strong', 'chase', 'protect']
CN = {'rae': 'Runs above expected under pressure (scoring only)', 'dc': 'Cost of getting out under pressure (expected future runs lost)', 'imp': 'Impact runs on pressure balls only = scoring + dismissal cost, each shrunk separately and added in runs', 'rae_n': 'Usual scoring level (calm balls)', 'dc_n': 'Usual dismissal cost (calm balls)', 'rae_d': 'Scoring: pressure minus usual (shrunk)', 'dc_d': 'Dismissal cost: pressure minus usual (shrunk)', 'imph': 'Impact under pressure = usual level + shrunk pressure deviation (scoring + dismissal cost, in runs)', 'oae': 'Dismissal avoided vs expected under pressure', 'waa': 'Win probability added above expected', 'slope_waa': 'Pressure slope (win impact rises with pressure)',
      'slope_iar': 'Pressure slope (impact rises with pressure)', 'hit': 'Consistency: share of pressure innings with positive win impact', 'floor': 'Downside: size of the bad pressure innings',
      'strong': 'Runs above expected vs strong sides (pressure balls)', 'chase': 'Runs above expected in chases (pressure balls)', 'protect': 'Strike protection with a tail-ender partner'}
# ---- PILLARS: overlapping components are averaged into one pillar so the same idea (runs above expected on overlapping balls) is not counted three times
PK = ['scoring', 'survival', 'slope', 'consistency', 'protect']
PNAME = {'scoring': 'Impact under pressure: usual level (all calm balls) plus the shrunk pressure deviation, counting the cost of getting out', 'survival': 'Survival and win impact above expected', 'slope': 'Pressure slope (rises with pressure vs own level)',
         'consistency': 'Consistency (hit rate and size of the bad innings)', 'protect': 'Strike protection with a tail-ender partner'}
PMEM = {'scoring': ['imph'], 'survival': ['oae', 'waa'], 'slope': ['slope_waa', 'slope_iar'], 'consistency': ['hit', 'floor'], 'protect': ['protect']}
G = np.zeros((len(PK), len(CK)))
for a_, p_ in enumerate(PK):
    for k_ in PMEM[p_]: G[a_, CK.index(k_)] = 1 / len(PMEM[p_])
FLOOR = np.array([0.05, 0.05, 0.10, 0.10, 0.05]); PCAP = 0.50
def estimates(T, sig):
    """(x, v) for every candidate component from additive totals T"""
    o = {}; W = np.maximum(T['W'], 1e-9)
    for k in ('rae', 'dc', 'oae', 'waa'):
        o[k] = (np.where(T['W'] > 0, T['P' + k] / W, np.nan), sig[k] * T['W2'] / W ** 2)
    for k in ('waa', 'iar'):
        sw = np.maximum(T['aW' + k], 1e-9); sxx_w = T['aXX' + k] - T['aX' + k] ** 2 / sw; sxr = T['aXR' + k] - T['aX' + k] * T['aR' + k] / sw
        sxx_u = T['uXX'] - T['uX'] ** 2 / np.maximum(T['uN'], 1)
        o['slope_' + k] = (np.where(sxx_w > 1e-6, sxr / np.maximum(sxx_w, 1e-9), np.nan), np.where(sxx_u > 1e-6, sig['slope_' + k] / np.maximum(sxx_u, 1e-9), np.nan))
    n = np.maximum(T['inn'], 1e-9); p = sig['hit_p']
    o['hit'] = (np.where(T['inn'] > 0, T['hit'] / n, np.nan), p * (1 - p) / n); o['floor'] = (np.where(T['inn'] > 0, T['flo'] / n, np.nan), sig['floor'] / n)
    Wn = np.maximum(T['nW'], 1e-9)
    for k_ in ('rae', 'dc'):
        xn = np.where(T['nW'] > 0, T['N' + k_] / Wn, np.nan); vn = sig[k_ + '_n'] * T['nW2'] / Wn ** 2; o[k_ + '_n'] = (xn, vn)
        o[k_ + '_d'] = (o[k_][0] - xn, o[k_][1] + vn)                     # pressure minus usual, with its sampling variance
    for tag, nm in (('s', 'strong'), ('c', 'chase')):
        Wt = np.maximum(T[tag + 'W'], 1e-9); o[nm] = (np.where(T[tag + 'W'] > 0, T[tag + 'R'] / Wt, np.nan), sig['rae'] * T[tag + 'W2'] / Wt ** 2)
    pn = np.maximum(T['pN'], 1e-9); o['protect'] = (np.where(T['pN'] > 0, T['pR'] / pn, np.nan), sig['protect'] / pn)
    return o
def components(T, sig, ok, fixed=None, rec=None):
    """fixed: hyper-parameters (and sampling variances) recorded on the full sample; rec: dict to record them into"""
    est = estimates(T, sig); Z = np.zeros((len(CK), NP)); S = np.full((len(CK), NP), np.nan); tau = np.zeros(len(CK)); B = np.zeros((len(CK), NP)); PM = {}
    COMB = {'imp': ('rae', 'dc'), 'imph': ('rae_n', 'dc_n', 'rae_d', 'dc_d')}
    for j, k in enumerate(CK):
        if k in COMB: continue
        x, v = est[k]; vv = fixed[k][2] if fixed else v
        pm_, ps, t, Bi, hp = eb(x, vv, ok, TAUD if k in ('rae_d', 'dc_d') else 0.0, (fixed[k][0], fixed[k][1]) if fixed else None); tau[j] = t; B[j] = Bi; PM[k] = (pm_, ps)
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
    out = np.full(NP, np.nan); g = ok & np.isfinite(x); out[g] = np.clip((x[g] - x[g].mean()) / (x[g].std() + 1e-12), -3, 3); return np.nan_to_num(out)
def target_Y(Tb, sig, ok):
    est = estimates(Tb, sig); return 0.5 * (raw_z(est['rae'][0] + est['dc'][0], ok) + raw_z(est['waa'][0], ok))
def spear(a, b, ok):
    idx = np.where(ok)[0]
    if np.std(a[idx]) < 1e-12 or np.std(b[idx]) < 1e-12: return 0.0          # a component shrunk entirely to zero carries no information
    return float(spearmanr(a[idx], b[idx])[0])
# ================= base build
thr_pressure = None
press0 = bb.press.values == 1
M0, SIG0 = build(press0, w_imp); Tfull = {k: v.sum(1) for k, v in M0.items()}
NP_MIN, PINN_MIN = 150, 12
ok_pop = (Tfull['n'] >= 25)
elig = (Tfull['n'] >= NP_MIN) & (Tfull['inn'] >= PINN_MIN) & active_p
elig_low = (Tfull['n'] >= 40) & (Tfull['inn'] >= 6) & active_p & (slot_p == 4)             # lower-order board has lower thresholds (tiny samples by nature)
elig = elig | elig_low
elig_all = ((Tfull['n'] >= NP_MIN) & (Tfull['inn'] >= PINN_MIN)) | ((Tfull['n'] >= 40) & (Tfull['inn'] >= 6) & (slot_p == 4))
print('pressure balls', int(press0.sum()), 'of', len(bb), '| players with >=25 pressure balls', int(ok_pop.sum()), '| eligible (active)', int(elig.sum()), '| by slot', {SN[k]: int((slot_p[elig] == k).sum()) for k in range(5)})
Z0, S0, TAU0, EST0, B0 = components(Tfull, SIG0, ok_pop); Zp0 = G @ Z0
print('tau (population signal sd) per component:', dict(zip(CK, np.round(TAU0, 4))))
print('median shrink factor (share of own data kept) among eligible:', {k: round(float(np.median(B0[j][elig])), 2) for j, k in enumerate(CK)})
# ================= validation: repeatability + validity on random halves (separate seeds for weight fitting / reporting) and chronological split
def halves(sel, mask=None):
    mk = np.ones(NM, bool) if mask is None else mask
    Ta = {k: v[:, sel & mk].sum(1) for k, v in M0.items()}; Tb = {k: v[:, ~sel & mk].sum(1) for k, v in M0.items()}
    return Ta, Tb, (Ta['n'] >= 25) & (Tb['n'] >= 25)
def run_halves(seeds, mask=None):
    rep = np.zeros(len(CK)); val = np.zeros(len(CK)); prep = np.zeros(len(PK)); pval = np.zeros(len(PK))
    for s_ in seeds:
        sel = np.random.default_rng(s_).random(NM) < .5; Ta, Tb, ok_ = halves(sel, mask)
        Za, _, _, _, _ = components(Ta, SIG0, ok_); Zb, _, _, _, _ = components(Tb, SIG0, ok_); Y = target_Y(Tb, SIG0, ok_); Pa = G @ Za; Pb = G @ Zb
        for j in range(len(CK)): rep[j] += spear(Za[j], Zb[j], ok_) / len(seeds); val[j] += spear(Za[j], Y, ok_) / len(seeds)
        for j in range(len(PK)): prep[j] += spear(Pa[j], Pb[j], ok_) / len(seeds); pval[j] += spear(Pa[j], Y, ok_) / len(seeds)
    return rep, val, prep, pval
rep_fit, val_fit, prep_fit, pval_fit = run_halves(range(100, 130))                         # used ONLY to choose weights
rep_rep, val_rep, prep_rep, pval_rep = run_halves(range(300, 330))                         # used ONLY for reporting
print('repeat (fit seeds)   ', dict(zip(CK, np.round(rep_fit, 2)))); print('validity (fit seeds) ', dict(zip(CK, np.round(val_fit, 2))))
print('repeat (report seeds)', dict(zip(CK, np.round(rep_rep, 2)))); print('validity (report seeds)', dict(zip(CK, np.round(val_rep, 2))))
print('PILLAR repeat (fit)', dict(zip(PK, np.round(prep_fit, 2))), '| validity (fit)', dict(zip(PK, np.round(pval_fit, 2))))
print('PILLAR repeat (report)', dict(zip(PK, np.round(prep_rep, 2))), '| validity (report)', dict(zip(PK, np.round(pval_rep, 2))))
def weights_from(val, rep):
    w = np.where(rep >= 0.03, np.maximum(val, 0.0), 0.0)            # a pillar must REPEAT (split-half >= 0.03) to earn weight beyond its floor
    w = w / w.sum() if w.sum() > 0 else np.ones(len(PK)) / len(PK)
    for _ in range(200): w = np.minimum(np.maximum(w, FLOOR), PCAP); w = w / w.sum()
    return w
WFIT = weights_from(pval_fit, prep_fit); WE = np.ones(len(PK)) / len(PK)
# CHOSEN WEIGHTS: only pillars with a rationale AND some measurable signal are scored. Slope ('vs own usual'), survival and strike protection were tested and showed no persistent
# player signal (split-half repeatability <= 0.03, slope negative), so scoring them would only add noise to a ranking whose purpose is reliability; they stay on the dashboard as information.
# Impact 70% / Consistency 30% (consistency is what the brief asks for; the data cannot separate 70/30 from 60/40, see the sensitivity variants).
WV = np.array([0.70, 0.0, 0.0, 0.30, 0.0]); WE = np.array([0.5, 0.0, 0.0, 0.5, 0.0])
print('fitted (validity-rule) weights for reference:', dict(zip(PK, np.round(WFIT, 3))), '| chosen:', dict(zip(PK, WV)))
print('pillar weights (validity-rule, reference only):', dict(zip(PK, np.round(WV, 3))))
LEVEL = np.array([1, 1, 0, 0, 0], float); LEVEL /= LEVEL.sum()
def design_scores(Za, Wv):
    Pa = G @ Za; P2 = Pa.copy(); P2[0] = (Za[CK.index('rae')] + Za[CK.index('chase')] + Za[CK.index('strong')]) / 3        # v3.0 scoring pillar: runs only, dismissal cost ignored
    return {'chosen weights (scoring + dismissal cost)': (Wv[:, None] * Pa).sum(0), 'chosen weights, scoring ignores dismissal cost': (Wv[:, None] * P2).sum(0), 'impact 50 / consistency 50': (WE[:, None] * Pa).sum(0), 'level only (scoring + survival)': (LEVEL[:, None] * Pa).sum(0)}
pred = {}
for s_ in range(300, 330):
    sel = np.random.default_rng(s_).random(NM) < .5; Ta, Tb, ok_ = halves(sel); Za, _, _, _, _ = components(Ta, SIG0, ok_); Zb, _, _, _, _ = components(Tb, SIG0, ok_); Y = target_Y(Tb, SIG0, ok_)
    for k_, v_ in design_scores(Za, WV).items(): pred.setdefault(k_, []).append(spear(v_, Y, ok_))
    pred.setdefault('_comp_repeat', []).append(spear((WV[:, None] * (G @ Za)).sum(0), (WV[:, None] * (G @ Zb)).sum(0), ok_))
date_by_mi = bb.groupby('mi' if 'mi' in bb else 'match_id').match_date.first(); dts = bb.groupby('match_id').match_date.first()
era = np.array([dts[x][:4] >= '2024' for x in sorted(bb.match_id.unique())]); era_res = {}
for lab, (fit_m, test_m) in {'2022-23 → 2024-26': (~era, era), '2024-26 → 2022-23': (era, ~era)}.items():
    Ta = {k: v[:, fit_m].sum(1) for k, v in M0.items()}; Tb = {k: v[:, test_m].sum(1) for k, v in M0.items()}; ok_ = (Ta['n'] >= 25) & (Tb['n'] >= 25)
    Za, _, _, _, _ = components(Ta, SIG0, ok_); Zf, _, _, _, _ = components({k: v[:, fit_m].sum(1) for k, v in M0.items()}, SIG0, ok_)
    # weights for this split come only from the fit period: validity of each component on random halves INSIDE the fit period
    vv = np.zeros(len(PK)); rr_ = np.zeros(len(PK))
    for s_ in range(500, 515):
        sel = np.random.default_rng(s_).random(NM) < .5; Ta2, Tb2, ok2 = halves(sel, fit_m); Za2, _, _, _, _ = components(Ta2, SIG0, ok2); Y2 = target_Y(Tb2, SIG0, ok2); P2 = G @ Za2
        Zb2, _, _, _, _ = components(Tb2, SIG0, ok2); Pb2 = G @ Zb2
        for j in range(len(PK)): vv[j] += spear(P2[j], Y2, ok2) / 15; rr_[j] += spear(P2[j], Pb2[j], ok2) / 15
    Wf = WV; Y = target_Y(Tb, SIG0, ok_)
    era_res[lab] = {k_: round(spear(v_, Y, ok_), 3) for k_, v_ in design_scores(Za, Wf).items()}; era_res[lab]['n'] = int(ok_.sum())
print('predictive validity (Spearman vs held-out pressure performance):', {k: round(float(np.mean(v)), 3) for k, v in pred.items()}); print('chronological:', era_res)
# ================= final composite
def composite(T, sig, ok, W, fixed=None, rec=None): Z, S, _, _, _ = components(T, sig, ok_pop, fixed, rec); return (W[:, None] * (G @ Z)).sum(0), Z
comp0, Zc = composite(Tfull, SIG0, ok_pop, WV)
eidx = np.where(elig)[0]; mu_c = comp0[eidx].mean(); sd_c = comp0[eidx].std()
nmap0 = {int(r.i): r.n for r in e[['striker_id', 'striker_name']].rename(columns={'striker_id': 'i', 'striker_name': 'n'}).drop_duplicates().itertuples()}
# ================= ENSEMBLE over defensible analyst choices, with the bootstrap run on the ENSEMBLE itself (so rank range / top-10 chance describe the ranking we publish)
rng = np.random.default_rng(7); BR = 300; DRAWS = np.stack([np.bincount(rng.integers(0, NM, NM), minlength=NM).astype(float) for _ in range(BR)])
ENS = [('Base definition (data decides how much pressure deviation counts)', press0, w_imp, WV, 0.0), ('Pressure = top 30% of the continuous pressure level', bb.pix.values >= np.quantile(bb.pix.values, 0.70), w_imp, WV, 0.0),
       ('Pressure = top 15% of the continuous pressure level', bb.pix.values >= np.quantile(bb.pix.values, 0.85), w_imp, WV, 0.0), ('No match-importance weighting', press0, w_one, WV, 0.0),
       ('Pressure-tilted: allow a modest real pressure effect (floor 0.04 runs/ball)', press0, w_imp, WV, 0.04)]
def zs(c): x = c[eidx]; return (x - x.mean()) / x.std()
ens_pt, ens_bt, sens = [], [], {}
for name, pr, w, W, taud in ENS:
    TAUD = taud; Mx, sx = (M0, SIG0) if name.startswith('Base') else build(pr, w)
    Tx = {k: v.sum(1) for k, v in Mx.items()}; REC = {}; c, Zx = composite(Tx, sx, ok_pop, W, rec=REC); zp = zs(c); ens_pt.append(zp); sens[name] = {'z': zp, 'ens': True}
    zb = np.zeros((BR, len(eidx)))
    for r in range(BR): cb, _ = composite({k: v @ DRAWS[r] for k, v in Mx.items()}, sx, ok_pop, W, fixed=REC); zb[r] = zs(cb)
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
    Mx, sx = build(press, w); Tx = {k: v.sum(1) for k, v in Mx.items()}; c, _ = composite(Tx, sx, ok_pop, WV if Wf is None else Wf); sens[name] = {'z': zs(c), 'ens': False}
addsens('Pressure = Pressure Index only (v2-style, top 20%; openers structurally excluded)', press=(bb.pi.values >= d['thr_pi']))
for j, k in enumerate(PK):
    if WV[j] > 0: wk = WV.copy(); wk[j] = 0; addsens('Without pillar: ' + PNAME[k], Wf=wk / wk.sum())
for nm_, wv_ in (('Impact 100% (no consistency)', [1, 0, 0, 0, 0]), ('Impact 60% / Consistency 40%', [0.6, 0, 0, 0.4, 0]), ('All five pillars equal (20% each)', [0.2] * 5)): addsens(nm_, Wf=np.array(wv_, float))
for v in sens.values(): o = np.argsort(-v['z']); v['o10'] = len(base10 & set(o[:10])); v['o30'] = len(base30 & set(o[:30]))
sens = {k: {'o10': v['o10'], 'o30': v['o30'], 'ens': v['ens']} for k, v in sens.items()}
print('sensitivity overlap with final top 10 / top 30:', {k: (v['o10'], v['o30']) for k, v in sens.items()})
# ---- calibration of the expected-outcome models by batting position and situation
cal_rows = lambda g: [[str(i), int(len(x)), round(100 * x.runs.mean(), 1), round(100 * x.e_runs.mean(), 1), round(100 * x.a_out.mean(), 2), round(100 * x.p_out.mean(), 2)] for i, x in g]
calib = {'order': cal_rows(bb.assign(o=bb.batting_order.clip(upper=9)).groupby('o')), 'situation': cal_rows(bb.assign(s=np.where(bb.press == 1, 'pressure', 'other')).groupby('s')), 'type': cal_rows(bb[bb.press == 1].assign(t=bb[bb.press == 1].ptype.map(dict(enumerate(TYPES)))).groupby('t')),
         'innings': cal_rows(bb.groupby('innings')), 'phase': cal_rows(bb.assign(p=np.where(bb.lb < 36, 'overs 1-6', np.where(bb.lb < 90, 'overs 7-15', 'overs 16-20'))).groupby('p'))}
print('calibration by batting position (n, SR actual/expected, out% actual/expected):', [(r[0], r[2], r[3], r[4], r[5]) for r in calib['order']])
# ================= player records
fb = bb.groupby('bat'); nmap = {int(r.i): r.n for r in e[['striker_id', 'striker_name']].rename(columns={'striker_id': 'i', 'striker_name': 'n'}).drop_duplicates().itertuples()}
tm = bb.groupby(['bat', 'bat_team']).size().reset_index(name='n').sort_values(['bat', 'n'], ascending=[True, False])
pos_all = {int(players[i]): k for k, i in enumerate(np.argsort(-comp0))}
def zst(df):
    n = len(df)
    if n == 0: return None
    return {'n': int(n), 'sr': round(100 * df.runs.mean(), 1), 'esr': round(100 * df.e_runs.mean(), 1), 'dot': round(100 * df.a_dot.mean(), 1), 'bnd': round(100 * df.a_bnd.mean(), 1), 'outs': int(df.a_out.sum()),
            'rae': round(100 * df.rae.mean(), 1), 'iar': round(100 * df.iar.mean(), 1), 'waa': round(float(df.waa.sum()), 2)}
ig_tab = bb[bb.press == 1].groupby('ig').agg(date=('match_date', 'first'), opp=('bowling_team_id', 'first'), runs=('runs', 'sum'), balls=('runs', 'size'), waa=('waa', 'sum'), won=('bat_won', 'first'), bat=('bat', 'first'))
ig_tab = ig_tab[ig_tab.balls >= 6]; igg = ig_tab.groupby('bat')
recs = []; eligpos = {int(eidx[i]): i for i in range(len(eidx))}
for p in fac[fac.balls >= 60].index:
    i = pidx[p]; df = fb.get_group(p); pr = df[df.press == 1]; nr = df[df.press == 0]; isel = i in eligpos; k = eligpos.get(i)
    tms = [tname.get(t, str(t)) for t in tm[tm.bat == p].bat_team.head(2)]
    rec = {'id': int(p), 'name': nmap.get(int(p), str(p)), 'teams': tms, 'slot': SN[slot_p[i]], 'balls': int(len(df)), 'elig': bool(isel), 'active': bool(active_p[i]), 'last': str(last_played.get(p, ''))[:10],
           'pn': int(Tfull['n'][i]), 'pinn': int(Tfull['inn'][i]), 'press': zst(pr), 'norm': zst(nr), 'ipl': round(float(df.is_ipl.mean()), 3),
           'types': {TYPES[t]: zst(pr[pr.ptype == t]) for t in range(len(TYPES)) if (pr.ptype == t).sum() >= 20}}
    if i in eligpos or (elig_all[i]):
        rec.update({'score': round(50 + 10 * ens_score[i], 1) if i in ens_score else round(50 + 10 * (float(comp0[i]) - mu_c) / sd_c, 1), 'score_base': round(50 + 10 * (float(comp0[i]) - mu_c) / sd_c, 1), 'ens': round(ens_score.get(i, float((comp0[i] - mu_c) / sd_c)), 3), 'rank_all': pos_all[int(p)] + 1, 'impS': round(100 * float(Zc[CK.index('imph')][i]) * float(TAU0[CK.index('imph')]), 1), 'z': [round(float(Zc[j][i]), 2) for j in range(len(CK))], 'zp': [round(float((G @ Zc)[j][i]), 2) for j in range(len(PK))],
                    'hit': round(float(Tfull['hit'][i] / max(Tfull['inn'][i], 1)), 2)})
    if isel:
        sdv = float(np.sqrt(np.nansum((WV * np.sqrt(np.maximum(G ** 2 @ np.nan_to_num(S0[:, i], nan=0.0) ** 2, 0))) ** 2)))     # posterior sd of the composite (independent-components approximation)
        rec.update({'rank': final_rank[i], 'medrank': float(med_rank[k]), 'rlo': int(rlo[k]), 'rhi': int(rhi[k]), 'top10p': round(float(top10p[k]), 3), 'sd': round(sdv / sd_c * 10, 2)})
        top = igg.get_group(p).sort_values('waa', ascending=False).head(3) if p in igg.groups else []
        rec['best'] = [[x.date, tname.get(x.opp, ''), int(x.runs), int(x.balls), int(x.won), round(float(x.waa), 2)] for x in top.itertuples()] if len(top) else []
    recs.append(rec)
meta = {'matches': int(bb.match_id.nunique()), 'balls': int(len(bb)), 'batters': NP, 'eligible': int(elig.sum()), 'pressShare': round(float(press0.mean()), 3), 'TYPES': TYPES, 'minPress': NP_MIN, 'minPinn': PINN_MIN, 'slots': SN, 'slotCounts': {SN[k]: int((slot_p[elig] == k).sum()) for k in range(5)},
        'calib': calib, 'pillars': {'keys': PK, 'fitted': [round(float(x), 3) for x in WFIT], 'names': [PNAME[k] for k in PK], 'weights': [round(float(x), 4) for x in WV], 'repeat': [round(float(x), 2) for x in prep_rep], 'validity': [round(float(x), 2) for x in pval_rep], 'members': PMEM},
        'comp': {'keys': CK, 'names': [CN[k] for k in CK], 'weights': [round(float(x), 4) for x in WV], 'tau': [round(float(x), 5) for x in TAU0], 'repeat': [round(float(x), 2) for x in rep_rep], 'validity': [round(float(x), 2) for x in val_rep],
                 'repeat_fit': [round(float(x), 2) for x in rep_fit], 'validity_fit': [round(float(x), 2) for x in val_fit], 'pred': {k: round(float(np.mean(v)), 3) for k, v in pred.items()}, 'era': era_res},
        'shareOfPressureByOrder': [[int(o), round(float(bb[bb.batting_order == o].press.mean()), 3), int((bb.batting_order == o).sum())] for o in range(1, 12)],
        'wp1': d['wp1'], 'wp2': d['wp2'], 'dq': d['dq'], 'dataEnd': m.match_date.max(), 'activeRule': 'IPL 2026 squad or a T20I in 2026', 'oppSource': d.get('oppSrc', '')}
json.dump({'meta': meta, 'players': recs, 'sens': sens}, open(OUT, 'w'), separators=(',', ':'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))
tops = sorted([r for r in recs if r['elig']], key=lambda r: r['rank'])
print('\nFINAL batters, consensus rank')
for r in tops[:30]: print(r['rank'], r['name'], '|', r['slot'][:8], '| score', r['score'], '±', r['sd'], '| medrank', r['medrank'], '[%d-%d]' % (r['rlo'], r['rhi']), 'top10p', r['top10p'], '| pn', r['pn'], 'pinn', r['pinn'], 'hit', r['hit'], '| pillars', r['zp'])
print('slots in top 10:', {SN[k]: sum(1 for r in tops[:10] if r['slot'] == SN[k]) for k in range(5)})
