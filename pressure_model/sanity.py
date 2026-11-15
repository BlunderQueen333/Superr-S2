"""Face-validity checks on the published lists: knock-out / final performance of the top 10, duplicate names, where well-known players sit.
usage: python sanity.py stage2.pkl bstage.pkl pressure_all.json bowlers_all.json"""
import sys, json, pickle, numpy as np, pandas as pd
d = pickle.load(open(sys.argv[1], 'rb')); bb = d['bb']; be = pickle.load(open(sys.argv[2], 'rb'))['e']
P = json.load(open(sys.argv[3])); B = json.load(open(sys.argv[4]))
top = lambda J, n=10: sorted([r for r in J['players'] if r['elig']], key=lambda r: r['rank'])[:n]
print('== KNOCK-OUT / FINAL games (stakes = 1) vs expected: batters')
ko = bb[bb.stakes == 1]
for r in top(P):
    x = ko[ko.bat == r['id']]
    print(f"{r['rank']:>2} {r['name']:<22} KO balls {len(x):>4} | SR {100*x.runs.mean() if len(x) else float('nan'):6.1f} vs exp {100*x.e_runs.mean() if len(x) else float('nan'):6.1f} | outs {int(x.a_out.sum()):>2} (exp {x.p_out.sum() if len(x) else 0:4.1f}) | impact/100 {100*x.iar.mean() if len(x) else float('nan'):6.1f}")
print('== KNOCK-OUT / FINAL games: bowlers')
bo = be[be.stakes == 1].assign(b=lambda t: pd.to_numeric(t.bowler_id, errors='coerce'))
for r in top(B):
    x = bo[bo.b == r['id']]; lb = x.legal.sum()
    print(f"{r['rank']:>2} {r['name']:<22} KO balls {int(lb):>4} | econ {6*x.ch.sum()/lb if lb else float('nan'):5.2f} vs exp {6*x.ech.sum()/lb if lb else float('nan'):5.2f} | wkts {int(x.cw.sum()):>2} (exp {x.pw.sum():4.1f})")
print('== duplicate names among all ranked players')
for nm, J in (('batters', P), ('bowlers', B)):
    names = [r['name'] for r in J['players'] if 'rank_all' in r]; dup = {n for n in names if names.count(n) > 1}; print(nm, 'duplicate names:', dup or 'none')
print('== where well-known batters sit (all-player pool)')
allb = {r['name']: r for r in P['players']}
for n in ['Virat Kohli', 'Rohit Sharma', 'Jos Buttler', 'Babar Azam', 'Suryakumar Yadav', 'Travis Head', 'Yashasvi Jaiswal', 'Phil Salt']:
    r = allb.get(n)
    print(n, '->', 'not in data (<60 balls)' if r is None else f"rank {r.get('rank', 'not ranked')} | pressure balls {r['pn']} | pressure innings {r['pinn']} | eligible {r['elig']}")
# ---- structured version for the report (worked examples + knock-out table)
out = {'bat': [], 'bowl': []}
for r in top(P, 12):
    bx = bb[bb.bat == r['id']]; g = lambda x: {'balls': int(len(x)), 'sr': float(100 * x.runs.mean()) if len(x) else None, 'esr': float(100 * x.e_runs.mean()) if len(x) else None, 'outs': int(x.a_out.sum()), 'eouts': float(x.p_out.sum()),
              'scoring': float(100 * x.rae.mean()) if len(x) else None, 'dismissal': float(100 * (x.iar - x.rae).mean()) if len(x) else None, 'impact': float(100 * x.iar.mean()) if len(x) else None}
    out['bat'].append({'name': r['name'], 'rank': r['rank'], 'press': g(bx[bx.press == 1]), 'calm': g(bx[bx.press == 0]), 'ko': g(bx[bx.stakes == 1])})
for r in top(B, 12):
    x = bo[bo.b == r['id']]; lb = float(x.legal.sum())
    out['bowl'].append({'name': r['name'], 'rank': r['rank'], 'ko': {'balls': int(lb), 'econ': float(6 * x.ch.sum() / lb) if lb else None, 'eecon': float(6 * x.ech.sum() / lb) if lb else None, 'wkts': int(x.cw.sum()), 'ewk': float(x.pw.sum())}})
json.dump(out, open('worked_example.json', 'w'))
