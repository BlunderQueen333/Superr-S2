"""Stage 2 (v3): pressure definition (index + slot-specific pressure TYPES) and slot-aware expected outcome for every ball faced.

Changes vs v2
  * Pressure Index re-weighted (leverage is the core; the other parts overlap with it so they carry less) and the 'game still alive' damping is smooth
  * pressure TYPES that exist from ball 1, so openers / #3 / tail are not structurally excluded:
        hard_chase   chasing a target >= 1.12 x venue-era par, first 8 overs
        early_wkt    a wicket in the last 3 overs while still inside the first 6 overs (platform has been hit)
        elite_new    new ball (first 6 overs) against a bowler who is stingier than 80% of bowlers
        rescue       3+ wickets down inside 12 overs, or 2+ down inside 5 overs
        last_pair    8+ wickets down with 2+ overs left
        index        top 20% of the Pressure Index (death overs, close finishes ...)
  * expected outcomes are conditioned on the batter's SLOT (batting order), the PARTNER (quality and whether the partner is a tail-ender, i.e. strike
    rotation is limited), target difficulty, team-strength gap, bowler type (leg-spin, left-arm ... ) and bowler speed
"""
import sys, pickle, os
PYLIB, IN, OUT, DATA = sys.argv[1:5]
if PYLIB != '-': sys.path.insert(0, PYLIB)
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from sklearn.model_selection import GroupKFold
d = pickle.load(open(IN, 'rb')); e = d['e']; mm_ = d['m']
HERE = os.path.dirname(os.path.abspath(__file__))
import official_strength as OS
teams_ = pd.read_csv(f'{DATA}/teams.csv'); z_ipl, z_icc, tn_, OPP_SRC, ICC_YEARS, _s1, _s2 = OS.build(mm_, teams_, HERE)
e['over'] = e.lb // 6
_bt = pd.to_numeric(e.bowling_team_id, errors='coerce')
e['opp_off'] = [z_ipl.get((c_, t_), 0.0) if i_ == 1 else z_icc.get((int(y_), tn_.get(t_)), 0.0) for c_, t_, i_, y_ in zip(e.comp_id, _bt, e.is_ipl, e.year)]
par_w = e.groupby('over').wk.mean(); e['par_w'] = e.over.map(par_w)
# ---- Pressure Index (0-100)
e['c_lev'] = e.lev_pct
e['c_dom'] = (0.6 * e.w18.clip(upper=3) / 3 + 0.4 * ((e.wk - e.par_w) / 3).clip(0, 1)).clip(0, 1)
req2 = ((e.rrr - e.par_rr) / 6).clip(0, 1)
req1 = (0.5 * ((0.5 - e.wp) / 0.25).clip(0, 1) + 0.5 * (e.dpar / 20).clip(0, 1))
e['c_req'] = np.where(e.innings == 2, req2, req1)
e['c_wk'] = 0.5 * e.c_dom + 0.5 * e.depth
e['c_mom'] = (0.5 * ((e.d12.fillna(4) - 4) / 6).clip(0, 1) + 0.5 * ((e.bsb - 6) / 12).clip(0, 1))
e['live'] = np.tanh(np.minimum(e.wp, 1 - e.wp) / 0.06)                    # smooth 'game still alive' damping (v2 used a hard clip at 10%)
PIW = {'c_lev': 0.40, 'c_req': 0.25, 'c_wk': 0.20, 'c_mom': 0.15}
e['pi_old'] = 100 * (PIW['c_lev'] * e.c_lev + e.live * (PIW['c_req'] * e.c_req + PIW['c_wk'] * e.c_wk + PIW['c_mom'] * e.c_mom))
# v4: pressure is defined ONCE, as leverage (the win-probability swing the ball can cause), damped when the game is already decided; no weighted blend of overlapping parts
e['pi'] = 100 * e.c_lev * (0.4 + 0.6 * e.live)
print('corr of PI parts', e[['c_lev', 'c_req', 'c_wk', 'c_mom']].corr().round(2).values.tolist())
# ---- faced balls
bb = e[e.extras_type != 'wide'].copy()
bb['bat'] = pd.to_numeric(bb.striker_id); bb['ns'] = pd.to_numeric(bb.non_striker_id)
bb['bf_before'] = bb.groupby(['match_id', 'innings', 'bat']).cumcount()
bb['pace'] = (bb.pace_or_spin == 'Pace').astype(int)
bb['out_s'] = bb.is_wicket & (bb.dismissed_player_id.astype(str) == bb.striker_id.astype(str))
bb['cls'] = np.select([bb.out_s, bb.runs == 0, bb.runs <= 3], [0, 1, 2], 3)
bb['maskw'] = ((~bb.extras_type.isin(['byes', 'legbyes'])) & ~(bb.is_wicket & ~bb.out_s)).astype(float)
bb['wpa_b'] = bb.wpa * bb.maskw
bb['bowler'] = pd.to_numeric(bb.bowler_id, errors='coerce')
mr = bb.groupby('over').runs.transform('mean'); mo = bb.groupby('over').out_s.transform('mean')
bb['rr'] = bb.runs - mr; bb['ro'] = bb.out_s.astype(float) - mo
def loo_eff(key, col, K, df=None):
    df = bb if df is None else df
    g_ = df.groupby([key]); s = g_[col].transform('sum'); n = g_[col].transform('size')
    gm = df.groupby([key, 'match_id']); sm = gm[col].transform('sum'); nm = gm[col].transform('size')
    return (s - sm) / (n - nm + K)
bb['bq_r'] = loo_eff('bowler', 'rr', 150); bb['bq_o'] = loo_eff('bowler', 'ro', 150); bb['vf'] = loo_eff('venue', 'rr', 400)
bb['opp'] = pd.to_numeric(bb.bowling_team_id, errors='coerce'); bb['opp_bq'] = loo_eff('opp', 'rr', 400)
tmr = e.drop_duplicates(['match_id', 'innings']).assign(opp=lambda x: pd.to_numeric(x.bowling_team_id, errors='coerce'), bw=lambda x: 1 - x.bat_won)[['match_id', 'opp', 'bw']]
tw = tmr.groupby('opp').bw.agg(['sum', 'size']); tmr = tmr.join(tw, on='opp')
tmr['opp_win'] = (tmr['sum'] - tmr.bw + 8 * 0.5) / (tmr['size'] - 1 + 8)
bb = bb.merge(tmr[['match_id', 'opp', 'opp_win']], on=['match_id', 'opp'], how='left')
# ---- PARTNER features: usual batting position and leave-one-match-out quality of the player at the other end
ord_med = bb.groupby('bat').batting_order.median()
bb['bat_q'] = loo_eff('bat', 'rr', 150)
pq = bb.groupby(['match_id', 'bat']).bat_q.first()
mi_ = pd.MultiIndex.from_arrays([bb.match_id.values, bb.ns.values])
bb['ns_q'] = pq.reindex(mi_).values
bb['ns_bo'] = bb.ns.map(ord_med).fillna(11).values
bb['ns_tail'] = (bb.ns_bo >= 8).astype(int)
bb['str_bo'] = bb.bat.map(ord_med).values
# ---- bowler type / speed
bt_codes = {k: i for i, k in enumerate(sorted(bb.bowling_type.astype(str).unique()))}
bb['btype'] = bb.bowling_type.astype(str).map(bt_codes).astype(int)
bb['bspeed'] = pd.to_numeric(bb.bowl_speed, errors='coerce').replace(0, np.nan)
# ---- pressure TYPES (v4). STRICT tier = headline definition; BROAD tier = the v3 definition, kept as a sensitivity check only
q20 = bb.bq_r.quantile(0.20)
live = (bb.wp.between(0.06, 0.94)).values
T = {}
T['last_pair'] = (bb.wk >= 8) & (bb.balls_left >= 12)
T['rescue'] = (((bb.wk >= 4) & (bb.lb < 72)) | ((bb.wk >= 3) & (bb.lb < 36))) & (bb.wp < 0.40)
T['early_wkt'] = (bb.lb < 36) & (bb.w18 >= 2) & (bb.wp < 0.45)
T['hard_chase'] = (bb.innings == 2) & (bb.tgt_ratio >= 1.15) & (bb.lb < 48) & (bb.wp < 0.45)
thr_pi = bb.pi.quantile(float(os.environ.get('STRICT_Q', 0.90))); T['index'] = bb.pi >= thr_pi
TYPES = ['last_pair', 'rescue', 'early_wkt', 'hard_chase', 'index']
ptype = np.full(len(bb), -1)
for k, nm in reversed(list(enumerate(TYPES))): ptype = np.where(T[nm].values & (live | (nm == 'index')), k, ptype)
bb['ptype'] = ptype; bb['press'] = (ptype >= 0).astype(int)
# broad tier (v3): types + elite_new + index top 20% of the OLD blended index
thr_old = bb.pi_old.quantile(0.80)
bb['press_broad'] = ((ptype >= 0) | (((bb.lb < 36) & (bb.bq_r <= q20)) & live) | (bb.pi_old >= thr_old) | (((bb.wk >= 3) & (bb.lb < 72) | (bb.wk >= 2) & (bb.lb < 30)) & (bb.wp < 0.45) & live)).astype(int)
pi_pct = bb.pi.rank(pct=True).values
bb['pix'] = np.maximum(pi_pct, np.where(ptype >= 0, 0.90, 0.0))
print('STRICT pressure share', round(bb.press.mean(), 3), '| broad share', round(bb.press_broad.mean(), 3), '| by type', {TYPES[k]: round(float((ptype == k).mean()), 3) for k in range(len(TYPES))})
bo = bb.groupby('batting_order').agg(n=('press', 'size'), pr=('press', 'mean')).head(11); print('pressure share by batting order', bo.pr.round(3).to_dict())
# ---- expected outcomes for the slot, partner, bowler and state (out-of-fold by match)
feats = ['innings', 'batting_order', 'wk', 'balls_left', 'need', 'rrr', 'crr', 'score', 'is_ipl', 'w18', 'bf_before', 'pace', 'wp',
         'par_rr', 'd12', 'bsb', 'fh', 'lhb', 'bq_r', 'bq_o', 'vf', 'comp3', 'opp_bq', 'opp_win', 'opp_off', 'dpar',
         'ns_q', 'ns_bo', 'ns_tail', 'tgt_ratio', 'tdiff', 'btype', 'bspeed']
cat = ['btype']
n = len(bb); P = np.zeros((n, 4)); Er = np.zeros(n); Ew = np.zeros(n); grp = bb.match_id.values
for tr, te in GroupKFold(5).split(bb, groups=grp):
    A = bb.iloc[tr]; B = bb.iloc[te]
    kw = dict(categorical_features=cat)
    P[te] = HGC(max_iter=170, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=0, **kw).fit(A[feats], A.cls).predict_proba(B[feats])
    Er[te] = HGR(max_iter=170, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=100, random_state=0, **kw).fit(A[feats], A.runs).predict(B[feats])
    Ew[te] = HGR(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=2.0, random_state=0, **kw).fit(A[feats], A.wpa_b).predict(B[feats]) * B.maskw.values
bb['p_out'], bb['p_dot'], bb['p_rot'], bb['p_bnd'] = P.T
bb['e_runs'] = Er; bb['e_wpa'] = Ew
bb['a_out'] = (bb.cls == 0).astype(int); bb['a_dot'] = (bb.cls == 1).astype(int); bb['a_rot'] = (bb.cls == 2).astype(int); bb['a_bnd'] = (bb.cls == 3).astype(int)
bb['rae'] = bb.runs - bb.e_runs; bb['oae'] = bb.p_out - bb.a_out; bb['waa'] = bb.wpa_b - bb.e_wpa     # all signed so that positive = better for the batter
print(bb[['p_out', 'a_out', 'p_dot', 'a_dot', 'e_runs', 'runs', 'e_wpa', 'wpa_b']].mean().round(4).to_dict())
# ---- IMPACT RUNS: runs scored + change in the batter's expected FUTURE runs, so a dismissal costs everything he was expected to score from there
# (a 6 off the first ball followed by a dismissal now nets strongly negative instead of a 'strike rate of 300')
bb = bb.sort_values(['match_id', 'innings', 'lb', 'bf_before']).reset_index(drop=True) if False else bb
g_ = bb.groupby(['match_id', 'innings', 'bat'], sort=False)
bb['fut'] = g_.runs.transform(lambda x: x[::-1].cumsum()[::-1])               # his runs from this ball to the end of his innings (this ball included)
Vp = np.zeros(len(bb))
for tr, te in GroupKFold(5).split(bb, groups=grp):
    Vp[te] = HGR(max_iter=170, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=100, random_state=0, categorical_features=cat).fit(bb.iloc[tr][feats], bb.iloc[tr].fut).predict(bb.iloc[te][feats])
bb['V'] = Vp; bb['V_next'] = g_.V.shift(-1).fillna(0.0)
bb['iar'] = bb.runs + bb.V_next - bb.V                                           # impact runs above expected (positive = better); mean ~ 0 by construction
print('impact runs: mean', round(float(bb.iar.mean()), 4), '| mean on pressure balls', round(float(bb[bb.press == 1].iar.mean()), 4), '| mean V at first ball', round(float(bb[bb.bf_before == 0].V.mean()), 1), '| mean cost of a dismissal', round(float(bb[bb.a_out == 1].iar.mean()), 1))
# ---- strike-protection data (all deliveries): set batter with a tail-ender at the other end
ee = e.copy(); ee['sid'] = pd.to_numeric(ee.striker_id, errors='coerce'); ee['nid'] = pd.to_numeric(ee.non_striker_id, errors='coerce')
ee['s_bo'] = ee.sid.map(ord_med).fillna(11); ee['n_bo'] = ee.nid.map(ord_med).fillna(11)
sp = ee[((ee.s_bo >= 8) ^ (ee.n_bo >= 8)) & (ee.lb >= 78) & (ee.legal == 1)].copy()          # exactly one tail-ender in the pair, overs 14-20
sp['setb'] = np.where(sp.s_bo < 8, sp.sid, sp.nid); sp['faced'] = (sp.s_bo < 8).astype(float); sp['over'] = sp.lb // 6
sp = sp.merge(bb[['match_id', 'innings', 'innings_ball_num', 'rae']], on=['match_id', 'innings', 'innings_ball_num'], how='left')
sp['rae_f'] = np.where(sp.faced == 1, sp.rae, np.nan)
sp = sp[['match_id', 'innings', 'setb', 'faced', 'over', 'wk', 'rae_f']]
print('strike-protection deliveries', len(sp), 'mean share faced by set batter', round(sp.faced.mean(), 3))
dc_tab = bb[['match_id', 'innings', 'innings_ball_num']].assign(wdc=-(bb.iar - bb.rae).values)      # value of this delivery's dismissal outcome to the BOWLING side (runs)
pickle.dump({'dc_tab': dc_tab, 'oppSrc': OPP_SRC, 'bb': bb, 'e': e, 'cal': d['cal'], 'm': d['m'], 'par_w': par_w, 'PIW': PIW, 'TYPES': TYPES, 'wp1': d['wp1'], 'wp2': d['wp2'], 'dq': d['dq'], 'ord_med': ord_med, 'sp': sp, 'thr_pi': float(thr_pi), 'thr_old': float(thr_old)}, open(OUT, 'wb'))
