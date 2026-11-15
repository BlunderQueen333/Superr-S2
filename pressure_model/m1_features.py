"""Stage 1 (v2): clean innings totals, build match state + context features, win-probability and leverage models.

Changes vs v1
  * target reconciled with the result text (v1 chases ended below target in ~27% of won chases)
  * venue/era par score (replaces the fixed 8 runs/over yardstick)
  * momentum: dots in last 12 balls, balls since last boundary
  * batting depth left (Impact Player era aware)
  * match stakes (knock-out / stage), competition level, venue id, free-hit flag
"""
import sys, re
PYLIB, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
if PYLIB != '-': sys.path.insert(0, PYLIB)
import os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
rd = lambda f: pd.read_csv(f'{DATA}/{f}', keep_default_na=False, na_values=[])
m = rd('matches.csv'); c = rd('competitions.csv'); b = rd('ball_by_ball.csv')
for col in ['runs', 'extras_2_runs', 'innings', 'overid', 'innings_ball_num', 'batting_order', 'match_id']:
    b[col] = pd.to_numeric(b[col], errors='coerce')
for col in ['is_wicket', 'is_four', 'is_six']:
    b[col] = b[col].astype(str).str.lower().eq('true')
m['match_id'] = pd.to_numeric(m.match_id)
m = m.merge(c[['comp_id', 'comp_name', 'match_format']], on='comp_id')
m['is_ipl'] = m.comp_name.str.startswith('IPL').astype(int)
m['comp3'] = np.where(m.is_ipl == 1, 0, np.where(m.match_format == 'T20I', 1, 2))
m['year'] = m.match_date.str[:4].astype(int)
m['dls'] = (m.revised_target.astype(str).str.len() > 0) | (m.first_inn_revised_overs.astype(str).str.len() > 0) | (m.second_inn_revised_overs.astype(str).str.len() > 0)
m['win'] = pd.to_numeric(m.win_team_id, errors='coerce')
m['stakes'] = np.where(pd.to_numeric(m.knock_out, errors='coerce') == 1, 1.0, np.where(m.stage == 'Super 4', 0.5, 0.0))
m['venue'] = pd.to_numeric(m.venue_id, errors='coerce')
# ---- qualification-race importance for multi-team tournaments (Asia Cup, tri-series, Emerging/Rising Stars cups, qualifiers)
def race_values(rows, k):
    """rows = chronological (match_id, home, away, win_team) of ONE group/table; k = places that qualify.
    per team: already clinched / already out -> 0.2 ; this result would clinch it (win) or eliminate it (loss) -> 1.0 ; otherwise live -> 0.65. match = mean of the two teams."""
    teams = {t for _, h, a, _ in rows for t in (h, a)}; pts = {t: 0 for t in teams}; rem = {t: 0 for t in teams}
    for _, h, a, _ in rows: rem[h] += 1; rem[a] += 1
    def clinched(p, x, t): return sum(1 for j in teams if j != t and x[j] >= p[t]) < k
    def out(p, x, t): return sum(1 for j in teams if j != t and p[j] > x[t]) >= k
    res = {}
    for mid, h, a, w in rows:
        mx = {t: pts[t] + 2 * rem[t] for t in teams}; vs = []
        for t, o in ((h, a), (a, h)):
            if clinched(pts, mx, t) or out(pts, mx, t): vs.append(0.2); continue
            pw = dict(pts); pw[t] += 2; xw = dict(mx); xw[o] -= 2
            pl = dict(pts); pl[o] += 2; xl = dict(mx); xl[t] -= 2
            vs.append(1.0 if (clinched(pw, xw, t) or out(pl, xl, t)) else 0.65)
        res[mid] = float(np.mean(vs))
        rem[h] -= 1; rem[a] -= 1
        if w in (h, a): pts[w] += 2
        else: pts[h] += 1; pts[a] += 1
    return res
def race_comp(g):
    """importance for the group-stage games of a multi-team tournament; detects a second round-robin phase (e.g. Super 4 stored as League)"""
    lg = g[g.stage.isin(['League', 'Others'])].sort_values(['match_date', 'mn']); rows = [(r.match_id, r.h, r.a, r.wt) for r in lg.itertuples()]
    has_next = g.stage.isin(['Final', 'Semi Final', 'Super 4']).any() or (pd.to_numeric(g.knock_out, errors='coerce') == 1).any()
    kq = 2 if has_next else 1; out = {}
    for n_ in range(len(rows) - 1, 5, -1):                     # second phase: last n_ games form a complete round robin of >= 4 teams that all played earlier
        suf, pre = rows[n_:], rows[:n_]
        st = {t for _, h, a, _ in suf for t in (h, a)}; pairs = {frozenset((h, a)) for _, h, a, _ in suf}
        if len(st) >= 4 and len(pairs) == len(suf) == len(st) * (len(st) - 1) // 2 and st <= {t for _, h, a, _ in pre for t in (h, a)}:
            out.update(race_values(suf, 2)); rows = pre; break
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for _, h, a, _ in rows: parent[find(h)] = find(a)
    for grp in {find(h) for _, h, _, _ in rows}:
        out.update(race_values([r for r in rows if find(r[1]) == grp], kq))
    return out
# ---- match importance (0..1) from tournament / series context
def match_importance(m):
    """knock-outs 0.9 (final 1.0), Super 4 0.7; IPL league = how tight the playoff race is for the two teams (a must-win scores 1.0,
    a team that is safe or already out 0.2, with 8 wins as the usual cut-off); bilateral series = same logic on 'wins still needed to clinch';
    other tournament group games 0.5"""
    mm = m.assign(mn=pd.to_numeric(m.match_number, errors='coerce'), h=pd.to_numeric(m.home_team_id, errors='coerce'), a=pd.to_numeric(m.away_team_id, errors='coerce'), wt=pd.to_numeric(m.win_team_id, errors='coerce')).sort_values(['comp_id', 'match_date', 'mn'])
    imp = {}
    tim = lambda need, rem: 0.2 if (need <= 0 or need > rem) else max(0.2, 1.0 / (1 + (rem - need)))
    for comp, g in mm.groupby('comp_id'):
        teams = set(g.h.dropna()) | set(g.a.dropna()); ipl = g.comp_name.iloc[0].startswith('IPL'); n = len(g)
        race = race_comp(g) if (not ipl and len(teams) > 2) else {}
        w = {t: 0 for t in teams}; gp = {t: 0 for t in teams}
        league_total = {t: int(((g.stage == 'League') & ((g.h == t) | (g.a == t))).sum()) for t in teams}
        for i, r in enumerate(g.itertuples()):
            if r.knock_out == 1 or str(r.knock_out) == '1': v = 1.0 if r.stage == 'Final' else 0.9
            elif r.stage == 'Super 4': v = 0.7
            elif r.match_id in race: v = race[r.match_id]
            elif ipl and r.stage == 'League':
                v = np.mean([tim(8 - w[t], league_total[t] - gp[t]) for t in (r.h, r.a) if t in w])
            elif (not ipl) and len(teams) == 2 and n >= 2:
                v = np.mean([tim(n // 2 + 1 - w[t], n - i) for t in (r.h, r.a) if t in w])
            else: v = 0.5
            imp[r.match_id] = float(v)
            for t in (r.h, r.a):
                if t in gp: gp[t] += 1
            wt = pd.to_numeric(r.win_team_id, errors='coerce')
            if wt in w: w[wt] += 1
    return imp
m['imp'] = m.match_id.map(match_importance(m)).fillna(0.5)
print('match importance by stage', m.groupby('stage').imp.mean().round(2).to_dict(), '| overall quantiles', m.imp.quantile([.1, .5, .9]).round(2).to_dict())
b = b[b.innings.isin([1, 2])].copy()
b = b.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)
b['e'] = np.where(b.extras_type == 'None', 0, np.where(b.extras_type == 'Penalty', 5, 1 + b.extras_2_runs))
b['tr'] = b.runs + b.e
b['legal'] = (~b.extras_type.isin(['wide', 'Overstep', 'other'])).astype(int)
# ---- innings totals: reconcile with the result text and ALLOCATE the missing (bye / leg-bye / wide-boundary) runs onto those balls
# (v2 only patched the first-innings total, so ball-level score / need / required rate drifted late in ~28% of chases)
tot = b.groupby(['match_id', 'innings']).tr.sum().unstack()
team2 = b[b.innings == 2].groupby('match_id').batting_team_id.first().astype(float)
mm = m.set_index('match_id')
T1c = tot[1]; T2c = tot[2]
bad = pd.Series(False, index=tot.index); status = pd.Series('as_recorded', index=tot.index); adj = {}
n_incons_before = 0; n_chase_won = 0
for mid in tot.index:
    if mid not in mm.index: continue
    r = mm.loc[mid]; res = str(r.result); w = r.win
    t1, t2 = T1c[mid], T2c[mid]
    if np.isnan(t1) or np.isnan(t2) or np.isnan(w): continue
    if w == team2.get(mid, np.nan):                       # chase won -> chaser must finish above the target
        n_chase_won += 1
        if t2 <= t1:
            n_incons_before += 1
            if t1 - t2 > 12: bad[mid] = True; status[mid] = 'incomplete_data'
            else: adj[(mid, 2)] = int(t1 + 1 - t2); status[mid] = 'chase_total_topped_up'
    else:
        mr = re.search(r'won by (\d+) runs?', res, re.I)
        if mr:                                            # bat-first win with exact margin -> t1 - t2 must equal the margin
            X = int(mr.group(1)); cand = t2 + X
            if abs(cand - t1) > 12: bad[mid] = True; status[mid] = 'incomplete_data'
            elif cand > t1: adj[(mid, 1)] = int(cand - t1); status[mid] = 'first_total_topped_up'
            elif cand < t1: adj[(mid, 2)] = int(t1 - X - t2); status[mid] = 'chase_total_topped_up'
b['_k'] = b.match_id.astype(np.int64) * 10 + b.innings.astype(np.int64)
alloc_left = 0; alloc_done = 0
for (mid, inn), d in adj.items():
    if d <= 0: continue
    for kinds in (['byes', 'legbyes'], ['wide', 'Overstep', 'other']):
        if d <= 0: break
        idx = b.index[(b.match_id == mid) & (b.innings == inn) & b.extras_type.isin(kinds)].to_numpy()
        if len(idx) == 0: continue
        add = np.zeros(len(idx), dtype=int); i = 0
        while d > 0 and add.min() < 4:            # one run at a time, round-robin, max 4 extra runs per ball
            if add[i % len(idx)] < 4: add[i % len(idx)] += 1; d -= 1
            i += 1
            if i > 8 * len(idx) + 8: break
        b.loc[idx, 'tr'] = b.loc[idx, 'tr'].to_numpy() + add; alloc_done += int(add.sum())
    alloc_left += max(d, 0)
tot = b.groupby(['match_id', 'innings']).tr.sum().unstack(); T1 = tot[1].copy(); T2c = tot[2]
b['T1'] = b.match_id.map(T1); b['T2'] = b.match_id.map(T2c)
print('chases won:', n_chase_won, 'recorded total <= target:', n_incons_before, '| reconciliation:', status.value_counts().to_dict(), '| runs allocated to extras balls', alloc_done, 'unallocated', alloc_left)
g = b.groupby(['match_id', 'innings'], sort=False)
b['score'] = g.tr.cumsum() - b.tr
b['wk'] = g.is_wicket.cumsum() - b.is_wicket.astype(int)
b['lb'] = g.legal.cumsum() - b.legal
b['balls_left'] = (120 - b.lb).clip(lower=0)
gid = g.ngroup().values
# per-innings rolling features (all use only balls BEFORE the current one)
w18 = np.zeros(len(b)); d12 = np.full(len(b), np.nan); bsb = np.zeros(len(b))
dotleg = ((b.legal == 1) & (b.tr == 0)).values.astype(int)
bnd = (b.is_four | b.is_six).values
for k in np.unique(gid):
    idx = np.where(gid == k)[0]; lb = b.lb.values[idx]; wk = b.wk.values[idx]
    j = np.searchsorted(lb, lb - 18, side='left'); w18[idx] = wk - wk[j]
    cd = np.cumsum(dotleg[idx]) - dotleg[idx]
    j = np.searchsorted(lb, lb - 12, side='left'); v = cd - cd[j]; v = v.astype(float); v[lb < 12] = np.nan; d12[idx] = v
    lastb = np.maximum.accumulate(np.where(bnd[idx], lb, -1)); prev = np.concatenate([[-1], lastb[:-1]])
    bsb[idx] = np.where(prev >= 0, lb - prev, lb)
b['w18'] = w18; b['d12'] = d12; b['bsb'] = np.minimum(bsb, 30)
b['fh'] = (b.groupby(['match_id', 'innings']).extras_type.shift(1) == 'Overstep').astype(int)
b['lhb'] = (b.batting_type == 'LHB').astype(int)
b = b.merge(m[['match_id', 'is_ipl', 'comp3', 'year', 'dls', 'win', 'match_date', 'comp_name', 'home_team_id', 'away_team_id', 'home_team', 'away_team', 'toss_win_team_id', 'knock_out', 'stakes', 'venue', 'stage', 'imp', 'comp_id']], on='match_id', how='left')
b['target'] = np.where(b.innings == 2, b.T1 + 1, np.nan)
b['need'] = b.target - b.score
b['rrr'] = np.where(b.innings == 2, (b.need * 6 / b.balls_left.clip(lower=1)).clip(upper=36), np.nan)
b['crr'] = np.where(b.lb > 0, b.score * 6 / b.lb.clip(lower=1), np.nan)
b['bat_team'] = pd.to_numeric(b.batting_team_id)
b['bat_won'] = (b.bat_team == b.win).astype(int)
ok = tot.dropna().index.difference(bad[bad].index)
b['elig'] = (~b.dls) & b.win.notna() & b.match_id.isin(ok)
# ---- venue / era par for the first innings (leave-one-out, shrunk to competition-year mean)
fl_ = (b[b.innings == 1].groupby('match_id').legal.sum() >= 108)
pm = m[['match_id', 'venue', 'comp3', 'year', 'dls']].copy(); pm['t1'] = pm.match_id.map(T1); pm['use'] = (~pm.dls) & pm.match_id.map(fl_).fillna(False) & pm.t1.notna() & ~pm.match_id.map(bad).fillna(False)
pm['u'] = pm.use.astype(float); pm['t1u'] = np.where(pm.use, pm.t1, 0.0)
def loo(keys, strength, prior):
    s = pm.groupby(keys).t1u.transform('sum'); n = pm.groupby(keys).u.transform('sum')
    s = s - pm.t1u; n = n - pm.u
    return (s + strength * prior) / (n + strength)
gmean = pm.t1u.sum() / pm.u.sum()
fam = loo(['comp3', 'year'], 5, gmean)
pm['par1'] = loo(['venue'], 6, fam).where(pm.venue.notna(), fam)
par1 = dict(zip(pm.match_id, pm.par1))
b['par1'] = b.match_id.map(par1); b['par_rr'] = b.par1 / 20
b['tgt_ratio'] = np.where(b.innings == 2, b.target / b.par1, np.nan)       # chase difficulty: target / venue-era par
# ---- pre-match team-strength prior (IPL end-of-season table / ICC rating, z-scored): a 60/3 chase is a different state for a strong and a weak side
import official_strength as OS
HERE = os.path.dirname(os.path.abspath(__file__))
z_ipl, z_icc, tn_, _src, _ya, _s1, _s2 = OS.build(m, rd('teams.csv'), HERE)
_bt = pd.to_numeric(b.bowling_team_id, errors='coerce')
zs = lambda c_, t_, i_, y_: z_ipl.get((c_, t_), 0.0) if i_ == 1 else z_icc.get((int(y_), tn_.get(t_)), 0.0)
b['tdiff'] = [zs(c_, bt_, i_, y_) - zs(c_, bw_, i_, y_) for c_, bt_, bw_, i_, y_ in zip(b.comp_id, b.bat_team, _bt, b.is_ipl, b.year)]
# cumulative par curve (share of a first-innings total scored by ball lb)
i1 = b[(b.innings == 1) & b.elig & (b.legal.sum() > 0)]
curve = (i1.groupby('lb').score.mean() / i1.groupby('match_id').T1.first().mean()).reindex(range(0, 121)).ffill().fillna(0)
b['dpar'] = (b.par1 * b.lb.map(curve) - b.score)                    # runs behind par (first innings)
b['depth_pos'] = np.where((b.is_ipl == 1) & (b.year >= 2023), 8, 7)    # Impact Player era: one more genuine batter
b['depth'] = 1 - ((b.depth_pos - (b.wk + 2)) / (b.depth_pos - 2)).clip(0, 1)
print('rows', len(b), 'eligible matches', b[b.elig].match_id.nunique(), 'dls', int(m.dls.sum()), 'bad-data matches', int(bad.sum()))
e = b[b.elig].copy()
def oof(model_fn, X, y, groups, cols, kind='clf'):
    p = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(X, y, groups):
        mdl = model_fn(); mdl.fit(X.iloc[tr][cols], y[tr]); p[te] = mdl.predict_proba(X.iloc[te][cols])[:, 1] if kind == 'clf' else mdl.predict(X.iloc[te][cols])
    return p
f2 = ['need', 'balls_left', 'wk', 'rrr', 'is_ipl', 'par_rr', 'd12', 'bsb', 'tdiff']; mono2 = [-1, 1, -1, -1, 0, 0, 0, 0, 0]
f1 = ['score', 'balls_left', 'wk', 'crr', 'is_ipl', 'par1', 'dpar', 'd12', 'tdiff']; mono1 = [1, 0, -1, 1, 0, 0, -1, 0, 0]
e['wp'] = np.nan
s = e.innings == 2; X = e[s]; y = X.bat_won.values
fn = lambda: HGC(max_iter=250, learning_rate=0.06, max_leaf_nodes=12, min_samples_leaf=60, l2_regularization=1.0, monotonic_cst=mono2, random_state=0)
p = oof(fn, X, y, X.match_id.values, f2); e.loc[s, 'wp'] = np.clip(p, 0.005, 0.995)
wp2 = {'brier': round(brier_score_loss(y, p), 4), 'base': round(brier_score_loss(y, np.full(len(y), y.mean())), 4), 'auc': round(roc_auc_score(y, p), 3)}
print('innings 2 WP', wp2)
first2 = e[e.innings == 2].groupby('match_id').wp.first()
e['soft'] = 1 - e.match_id.map(first2)
s1 = e.innings == 1; X = e[s1]; p = np.zeros(len(X))
for tr, te in GroupKFold(5).split(X, groups=X.match_id.values):
    mdl = HGR(max_iter=250, learning_rate=0.06, max_leaf_nodes=12, min_samples_leaf=60, l2_regularization=1.0, monotonic_cst=mono1, random_state=0)
    mdl.fit(X.iloc[tr][f1], X.iloc[tr].soft); p[te] = mdl.predict(X.iloc[te][f1])
e.loc[s1, 'wp'] = np.clip(p, 0.005, 0.995)
yy = X.bat_won.values
wp1 = {'auc': round(roc_auc_score(yy, p), 3)}
print('innings 1 WP', wp1)
e['wp_after'] = e.groupby(['match_id', 'innings']).wp.shift(-1)
last = e.groupby(['match_id', 'innings']).cumcount(ascending=False) == 0
l1 = last & (e.innings == 1); l2 = last & (e.innings == 2)
e.loc[l1, 'wp_after'] = 1 - e.loc[l1, 'match_id'].map(first2)
e.loc[l2, 'wp_after'] = e.loc[l2, 'bat_won']
e['wpa'] = e.wp_after - e.wp
cal = {}
for inn in (1, 2):
    s_ = e[e.innings == inn]; q = pd.cut(s_.wp, np.linspace(0, 1, 11), include_lowest=True)
    t = s_.groupby(q, observed=True).agg(p=('wp', 'mean'), a=('bat_won', 'mean'), n=('wp', 'size'))
    cal[inn] = [[round(r.p, 4), round(r.a, 4), int(r.n)] for r in t.itertuples()]
fl = ['innings', 'need', 'balls_left', 'wk', 'rrr', 'crr', 'score', 'is_ipl', 'par_rr', 'd12', 'bsb', 'tdiff']
e['absw'] = e.wpa.abs()
p = np.zeros(len(e))
for tr, te in GroupKFold(5).split(e, groups=e.match_id.values):
    mdl = HGR(max_iter=200, learning_rate=0.06, max_leaf_nodes=12, min_samples_leaf=80, random_state=0)
    mdl.fit(e.iloc[tr][fl], e.iloc[tr].absw); p[te] = mdl.predict(e.iloc[te][fl])
e['lev'] = p; e['lev_pct'] = e.lev.rank(pct=True)
import pickle
dq = {'chases_won': int(n_chase_won), 'target_below_recorded_chase': int(n_incons_before), 'recon': {k: int(v) for k, v in status.value_counts().items()}, 'bad_matches': int(bad.sum())}
pickle.dump({'b': b, 'e': e, 'cal': cal, 'm': m, 'wp1': wp1, 'wp2': wp2, 'dq': dq}, open(OUT, 'wb'))
print(e[['wp', 'wpa', 'lev']].describe())
