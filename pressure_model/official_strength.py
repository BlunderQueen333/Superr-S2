"""Official opponent strength - PRE-MATCH (no hindsight): a match uses the PREVIOUS IPL season's final table and the PREVIOUS year-end ICC table
(first IPL season: neutral; 2022 internationals: the 2022 year-end table is the only one available).
IPL: end-of-season league table (points z-score within the season), built from the match results.
Internationals: ICC men's T20I team rating, z-scored. If icc_t20i_ratings_by_year.csv (columns: year, team, rating = ICC rating on 31 Dec of that year)
exists it is used for each match's calendar year (2026 uses the latest year present); teams missing for a year fall back to the snapshot in icc_t20i_ratings.csv.
"""
import os, numpy as np, pandas as pd
def build(m, teams, here):
    tn = dict(zip(teams.team_id, teams.team_name))
    snap = pd.read_csv(os.path.join(here, 'icc_t20i_ratings.csv')).set_index('team').rating.to_dict()
    p = os.path.join(here, 'icc_t20i_ratings_by_year.csv'); byy = pd.read_csv(p) if os.path.exists(p) else None
    years_avail = sorted(byy.year.unique()) if byy is not None else []
    lookup = {(int(r.year), r.team): float(r.rating) for r in byy.itertuples()} if byy is not None else {}
    def rating(name, year):
        is_a = name.endswith(' A') and name[:-2] in snap
        base = name[:-2] if is_a else name
        v = lookup.get((int(year), base))
        if v is None: v = snap.get(base, np.nan)          # no year-end value supplied: use the 1 Nov 2026 snapshot
        return v - (40.0 if is_a else 0.0)
    ipl_kw = ['Super Kings', 'Indians', 'Royals', 'Punjab', 'Bengaluru', 'Capitals', 'Knight', 'Sunrisers', 'Titans', 'Giants']
    names = sorted({tn[t] for t in teams.team_id.unique() if tn.get(t) is not None and not any(k in tn[t] for k in ipl_kw)})
    z_icc = {}
    for y in sorted(set(m.year.astype(int))):
        r = np.array([rating(n, y) for n in names]); mu, sd = np.nanmean(r), np.nanstd(r)
        for n, v in zip(names, r): z_icc[(y, n)] = 0.0 if np.isnan(v) else (v - mu) / sd
    s_icc = {}
    for y in sorted(set(m.year.astype(int))):
        rk = sorted(names, key=lambda n: -(rating(n, y) if not np.isnan(rating(n, y)) else -1e9))
        for i_, n in enumerate(rk): s_icc[(y, n)] = i_ + 1          # ICC rank that year (strong = rank <= k, default 5)
    z_ipl = {}; s_ipl = {}
    for comp, g in m[(m.is_ipl == 1) & (m.stage == 'League')].groupby('comp_id'):
        pts = {}
        for r in g.itertuples():
            for t in (pd.to_numeric(r.home_team_id, errors='coerce'), pd.to_numeric(r.away_team_id, errors='coerce')): pts[t] = pts.get(t, 0) + (1 if pd.isna(r.win) else 0)
            if not pd.isna(r.win): pts[r.win] = pts.get(r.win, 0) + 2
        v = np.array(list(pts.values()), float)
        for t, p_ in pts.items(): z_ipl[(comp, t)] = (p_ - v.mean()) / (v.std() + 1e-9)
        for i_, t in enumerate(sorted(pts, key=lambda t: -pts[t])): s_ipl[(comp, t)] = i_ + 1          # IPL league finish that season (strong = rank <= k, default 4)
    # ---- make everything PRE-MATCH: IPL uses last season's table, internationals use last year-end's ICC table
    ipl_year = m[m.is_ipl == 1].groupby('comp_id').year.first().astype(int).to_dict(); by_year = {y: c for c, y in ipl_year.items()}
    z_ipl_l, s_ipl_l = {}, {}
    for (comp, t) in z_ipl:
        prev = by_year.get(ipl_year[comp] - 1)
        z_ipl_l[(comp, t)] = z_ipl.get((prev, t), 0.0) if prev is not None else 0.0
        s_ipl_l[(comp, t)] = s_ipl.get((prev, t), 99) if prev is not None else 99
    ymin = min(int(y) for y, _ in z_icc); z_icc_l = {(y, n): z_icc[(max(y - 1, ymin), n)] for (y, n) in z_icc}; s_icc_l = {(y, n): s_icc[(max(y - 1, ymin), n)] for (y, n) in s_icc}
    z_ipl, s_ipl, z_icc, s_icc = z_ipl_l, s_ipl_l, z_icc_l, s_icc_l
    yrs_in_data = sorted(set(m.year.astype(int))); miss = [y for y in yrs_in_data if y not in years_avail]
    src = ("IPL end-of-season league tables + ICC men's T20I ratings (top 11 on 31 Dec of " + ', '.join(str(int(y)) for y in years_avail) + "; the 1 Nov 2026 snapshot for " + ', '.join(str(y) for y in miss) + " and for teams outside the top 11)") if years_avail else 'IPL end-of-season league tables + ICC men\'s T20I ratings (1 Nov 2026 snapshot for every year)'
    return z_ipl, z_icc, tn, src, years_avail, s_ipl, s_icc
