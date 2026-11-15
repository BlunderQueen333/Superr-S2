"""Bowling stage 1 (v3): expected outcome per delivery + bowling pressure definition aligned with the batting one.

Pressure core = the SAME win-probability / leverage model as batting; bowler-specific states are added on top:
  Bowling Pressure Index = 100 x [0.50 leverage + live x (0.30 knife-edge + 0.20 under-attack)]       (v2 had 30% 'late in the innings', which favoured death bowlers by construction)
  pressure TYPES (primary type = first that applies):
     death_close   overs 16-20 with the result still open (win chance 25-75%)
     under_attack  the batting side has scored >= 22 off the last 12 deliveries (10+ an over) and the game is live (bowler is being hit)
     knife_edge    win chance within 43-57% (any phase)
     index         top 15% of the Bowling Pressure Index
Expected values now know: bowler TYPE (leg-spin, left-arm, fast-medium ...) and speed, batter quality/hand, target difficulty, team-strength gap, venue.
Extra signals: boundary prevention and discipline (wides / no-balls) against expectation.
"""
import sys, pickle, os
PYLIB, IN, OUT, DATA = sys.argv[1:5]
if PYLIB != '-': sys.path.insert(0, PYLIB)
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from sklearn.model_selection import GroupKFold
import official_strength as OS
d = pickle.load(open(IN, 'rb')); e = d['e'].copy(); m = d['m']
e = e.merge(d['dc_tab'], on=['match_id', 'innings', 'innings_ball_num'], how='left'); e['wdc'] = e.wdc.fillna(0.0)      # wicket value in runs (0 on wides: no batter faced)
HERE = os.path.dirname(os.path.abspath(__file__))
teams = pd.read_csv(f'{DATA}/teams.csv')
z_ipl, z_icc, tn, src, ya, s_ipl, s_icc = OS.build(m, teams, HERE)
bt = pd.to_numeric(e.batting_team_id, errors='coerce')
e['opp_off_bat'] = [z_ipl.get((c, t), 0.0) if i == 1 else z_icc.get((int(y), tn.get(t)), 0.0) for c, t, i, y in zip(e.comp_id, bt, e.is_ipl, e.year)]
e['strong_bat'] = [(s_ipl.get((c, t), 99) <= 4) if i == 1 else (s_icc.get((int(y), tn.get(t)), 99) <= 4) for c, t, i, y in zip(e.comp_id, bt, e.is_ipl, e.year)]
# ---- what the bowler is charged / credited
e['over'] = e.lb // 6
e['ch'] = np.where(e.extras_type.isin(['byes', 'legbyes', 'Penalty']), 0, np.where(e.extras_type.isin(['wide', 'Overstep', 'other']), e.runs + e.e, e.runs)).astype(float)
e['cw'] = (e.is_wicket & e.dismissal_type.isin(['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket'])).astype(int)
e['dotb'] = ((e.legal == 1) & (e.ch == 0)).astype(int)
e['bnd'] = ((e.is_four | e.is_six) & ~e.extras_type.isin(['byes', 'legbyes'])).astype(int)
e['xt'] = e.extras_type.isin(['wide', 'Overstep']).astype(int)                   # wides and no-balls (free hits follow)
mask_w = ((~e.extras_type.isin(['byes', 'legbyes'])) & ~(e.is_wicket & (e.cw == 0))).astype(float)
e['bw_mask'] = mask_w; e['bwpa'] = -e.wpa * mask_w
e['pace'] = (e.pace_or_spin == 'Pace').astype(int)
bt_codes = {k: i for i, k in enumerate(sorted(e.bowling_type.astype(str).unique()))}
e['btype'] = e.bowling_type.astype(str).map(bt_codes).astype(int); e['bspeed'] = pd.to_numeric(e.bowl_speed, errors='coerce').replace(0, np.nan)
g = e.groupby(['match_id', 'innings'], sort=False)
e['rr12'] = g.tr.transform(lambda x: x.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
# ---- batter on strike: quality (leave-one-match-out, phase adjusted), balls faced so far
e['strk'] = pd.to_numeric(e.striker_id, errors='coerce')
f = e[e.extras_type != 'wide'].copy()
f['rr'] = f.runs - f.groupby('over').runs.transform('mean')
f['os'] = (f.is_wicket & (f.dismissed_player_id.astype(str) == f.striker_id.astype(str))).astype(float); f['ro'] = f.os - f.groupby('over').os.transform('mean')
def loo(col, key, K):
    g_ = f.groupby(key); s = g_[col].transform('sum'); n = g_[col].transform('size'); gm = f.groupby([key, 'match_id']); return (s - gm[col].transform('sum')) / (n - gm[col].transform('size') + K)
f['sq_r'] = loo('rr', 'strk', 150); f['sq_o'] = loo('ro', 'strk', 150); f['vf'] = loo('rr', 'venue', 400)
q = f.groupby(['strk', 'match_id'])[['sq_r', 'sq_o']].first().reset_index(); v = f.groupby(['venue', 'match_id'])[['vf']].first().reset_index()
e = e.merge(q, on=['strk', 'match_id'], how='left').merge(v, on=['venue', 'match_id'], how='left')
e['fl'] = (e.extras_type != 'wide').astype(int); e['bat_bf'] = e.groupby(['match_id', 'innings', 'strk']).fl.cumsum() - e.fl
# ---- expected outcomes, out-of-fold by match
FB = ['innings', 'over', 'wk', 'balls_left', 'need', 'rrr', 'crr', 'score', 'is_ipl', 'w18', 'd12', 'bsb', 'wp', 'par_rr', 'fh', 'lhb', 'pace', 'sq_r', 'sq_o', 'bat_bf', 'batting_order', 'vf', 'comp3', 'opp_off_bat', 'dpar',
      'tgt_ratio', 'tdiff', 'btype', 'bspeed', 'rr12']
n = len(e); Ech = np.zeros(n); Pw = np.zeros(n); Pd = np.zeros(n); Ebw = np.zeros(n); Pb = np.zeros(n); Px = np.zeros(n); grp = e.match_id.values
kw = dict(categorical_features=['btype'])
for tr, te in GroupKFold(5).split(e, groups=grp):
    A = e.iloc[tr]; B = e.iloc[te]; L = A[A.legal == 1]
    Ech[te] = HGR(max_iter=170, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=100, random_state=0, **kw).fit(A[FB], A.ch).predict(B[FB])
    Pw[te] = HGC(max_iter=170, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=1.0, random_state=0, **kw).fit(A[FB], A.cw).predict_proba(B[FB])[:, 1]
    Pd[te] = HGC(max_iter=170, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=0, **kw).fit(L[FB], L.dotb).predict_proba(B[FB])[:, 1]
    Pb[te] = HGC(max_iter=170, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=0, **kw).fit(L[FB], L.bnd).predict_proba(B[FB])[:, 1]
    Px[te] = HGC(max_iter=120, learning_rate=0.07, max_leaf_nodes=12, min_samples_leaf=150, l2_regularization=1.0, random_state=0, **kw).fit(A[FB], A.xt).predict_proba(B[FB])[:, 1]
    Ebw[te] = HGR(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=2.0, random_state=0, **kw).fit(A[FB], A.bwpa).predict(B[FB]) * B.bw_mask.values
e['ech'] = Ech; e['pw'] = Pw; e['pd'] = np.where(e.legal == 1, Pd, 0.0); e['pbnd'] = np.where(e.legal == 1, Pb, 0.0); e['pxt'] = Px; e['ebw'] = Ebw
lg = e.legal == 1
print('check (actual vs expected):', {k: round(float(a), 4) for k, a in [('charged runs', e.ch.mean()), ('exp', e.ech.mean()), ('wicket', e.cw.mean()), ('exp', e.pw.mean()), ('dot', e[lg].dotb.mean()), ('exp', e[lg].pd.mean()), ('boundary', e[lg].bnd.mean()), ('exp', e[lg].pbnd.mean()), ('wide/nb', e.xt.mean()), ('exp', e.pxt.mean())]})
# ---- Bowling Pressure Index and types
e['closeness'] = 1 - np.clip(np.abs(e.wp - 0.5) / 0.5, 0, 1)
e['attack'] = np.clip((e.rr12 - 16) / 12, 0, 1)
BPIW = {'c_lev': 0.50, 'closeness': 0.30, 'attack': 0.20}
e['bpi'] = 100 * (BPIW['c_lev'] * e.c_lev + e.live * (BPIW['closeness'] * e.closeness + BPIW['attack'] * e.attack))
live = e.wp.between(0.06, 0.94)
thr = e.bpi.quantile(0.90)
# v4 STRICT tier: knife_edge only counts in the CHASE (first-innings win probability sits near 50% by construction early on, which made the powerplay look 'high pressure'); index = top 10%
T = {'death_close': (e.over >= 15) & e.wp.between(0.30, 0.70), 'under_attack': (e.rr12 >= 26) & live & (e.over >= 2), 'knife_edge': e.wp.between(0.43, 0.57) & (e.innings == 2) & (e.over >= 2), 'index': e.bpi >= thr}
BT = ['death_close', 'under_attack', 'knife_edge', 'index']
pt = np.full(len(e), -1)
for k, nm in reversed(list(enumerate(BT))): pt = np.where(T[nm].values, k, pt)
# v3 BROAD tier (sensitivity only)
Tb = {'death_close': (e.over >= 15) & e.wp.between(0.25, 0.75), 'under_attack': (e.rr12 >= 22) & live & (e.over >= 2), 'knife_edge': e.wp.between(0.43, 0.57), 'index': e.bpi >= e.bpi.quantile(0.85)}
e['press_broad'] = (np.logical_or.reduce([v.values for v in Tb.values()])).astype(int)
e['ptype'] = pt; e['press'] = (pt >= 0).astype(int)
e['bpix'] = np.maximum(e.bpi.rank(pct=True).values, np.where(pt >= 0, 0.90, 0.0))
print('pressure share', round(e.press.mean(), 3), {BT[k]: round(float((pt == k).mean()), 3) for k in range(len(BT))}, '| by over', e.groupby('over').press.mean().round(2).to_dict())
pr = e.press == 1
print('pressure vs other: runs/over', round(6 * e[pr].ch.sum() / e[pr].legal.sum(), 2), round(6 * e[~pr].ch.sum() / e[~pr].legal.sum(), 2), '| wickets/over', round(6 * e[pr].cw.sum() / e[pr].legal.sum(), 3), round(6 * e[~pr].cw.sum() / e[~pr].legal.sum(), 3), '| fielding-side win%', round(100 * (1 - e[pr].bat_won.mean()), 1))
keep = ['press_broad', 'innings_ball_num', 'sq_r', 'strk', 'dismissed_player_id', 'venue', 'match_id', 'innings', 'lb', 'wdc', 'over', 'bowler_id', 'bowler_name', 'bowling_team_id', 'batting_team_id', 'pace_or_spin', 'bowling_type', 'batting_type', 'legal', 'ch', 'cw', 'dotb', 'bnd', 'xt', 'bw_mask', 'bwpa', 'ech', 'pw', 'pd', 'pbnd', 'pxt', 'ebw',
        'bpi', 'bpix', 'ptype', 'press', 'strong_bat', 'opp_off_bat', 'imp', 'is_ipl', 'match_date', 'year', 'comp_id', 'comp_name', 'bat_won', 'stakes', 'wp', 'batting_order']
pickle.dump({'e': e[keep].copy(), 'm': m, 'BPIW': BPIW, 'BT': BT}, open(OUT, 'wb'))
