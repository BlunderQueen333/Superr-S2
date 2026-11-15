"""Builds the methodology report (Markdown + Word) and its charts from batters_v4.json / bowlers_v4.json.   usage: python make_report.py [outdir]"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'report_v4'); os.makedirs(OUT, exist_ok=True)
B = json.load(open(os.path.join(HERE, 'batters_v4.json'))); W = json.load(open(os.path.join(HERE, 'bowlers_v4.json')))
bm, wm = B['meta'], W['meta']
bt = sorted([p for p in B['players'] if p['elig']], key=lambda p: p['rank']); wt = sorted([p for p in W['players'] if p['elig']], key=lambda p: p['rank'])
BLUE, ORANGE, INK, MUTED, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d9d8d3', '#fcfcfb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .5, 'axes.axisbelow': True})
def save(fig, name): p = os.path.join(OUT, name); fig.savefig(p, dpi=160, bbox_inches='tight'); plt.close(fig); return p

# ---------- fig 1: pressure vs ordinary (small multiples, one scale per panel)
def fig_tiers():
    fig, ax = plt.subplots(2, 4, figsize=(9.5, 4.6))
    spec = [(bm, ['strike rate', 'dismissal % per ball', 'dot %', 'sd of win-prob swing per ball (x100)'], 'Batters'), (wm, ['economy', 'wickets per over', 'boundary %', 'dot %'], 'Bowlers')]
    for r, (m, keys, lab) in enumerate(spec):
        for c, k in enumerate(keys):
            a = ax[r, c]; v = [m['tiers']['Pressure (strict tier)'][k], m['tiers']['Ordinary'][k]]
            b = a.bar([0, 1], v, color=[BLUE, ORANGE], width=.62)
            for x, y in zip([0, 1], v): a.text(x, y, f'{y:g}', ha='center', va='bottom', color=INK, fontsize=8)
            a.set_xticks([0, 1]); a.set_xticklabels(['Pressure', 'Ordinary'], fontsize=8); a.set_title(k.replace(' (x100)', ''), fontsize=8.5, color=INK); a.set_ylim(0, max(v) * 1.2); a.grid(False); a.yaxis.set_visible(False); a.spines['left'].set_visible(False)
        ax[r, 0].set_ylabel(lab, color=INK, fontweight='bold', fontsize=10); ax[r, 0].yaxis.set_visible(False)
    fig.suptitle('Strict pressure tier vs ordinary balls (same players, same matches)', color=INK, fontsize=10.5, y=1.0); fig.tight_layout(); return save(fig, 'fig1_pressure_vs_ordinary.png')
# ---------- fig 2/3: rank ranges
def fig_rank(pl, name, title):
    top = pl[:15]; fig, a = plt.subplots(figsize=(7.2, 4.8))
    for i, p in enumerate(top):
        y = len(top) - 1 - i; a.plot([p['rlo'], p['rhi']], [y, y], color=BLUE if p['rank'] <= 10 else MUTED, lw=2.2, alpha=.85, solid_capstyle='round'); a.plot(p['medrank'], y, 'o', color=INK, ms=5, mec=SURF, mew=1.5)
        a.text(max(p['rhi'], 1) + 1.2, y, f"P(top-10) {p['top10p']:.0%}" + ('  low sample' if p['lowconf'] else ''), va='center', fontsize=7.5, color=MUTED)
    a.set_yticks(range(len(top))); a.set_yticklabels([f"{p['rank']}. {p['name']}" for p in top][::-1], fontsize=8.5); a.axvline(10.5, color=ORANGE, lw=1, ls='--'); a.text(10.8, len(top) - .6, 'top-10 line', color=ORANGE, fontsize=7.5)
    a.set_xlabel('Rank across 400 stress draws (bar = 5th-95th percentile, dot = median)'); a.set_title(title, color=INK, fontsize=10.5, loc='left'); a.grid(axis='y', visible=False); a.set_xlim(0, max(p['rhi'] for p in top) + 25); fig.tight_layout(); return save(fig, name)
# ---------- fig 4: pressure vs normal (batters)
def fig_scatter():
    el = [p for p in B['players'] if p['elig'] and p['press'] and p['norm']]; fig, a = plt.subplots(figsize=(5.6, 5.2))
    x = np.array([p['norm']['rae'] for p in el]); y = np.array([p['press']['rae'] for p in el]); top = {p['id'] for p in bt[:10]}
    a.scatter(x, y, s=22, color=MUTED, alpha=.55, ec=SURF, lw=.8, label='Other eligible batters'); a.scatter([p['norm']['rae'] for p in el if p['id'] in top], [p['press']['rae'] for p in el if p['id'] in top], s=40, color=BLUE, ec=SURF, lw=1.2, label='Final top 10')
    lim = [min(x.min(), y.min()) - 5, max(x.max(), y.max()) + 5]; a.plot(lim, lim, color=ORANGE, lw=1.2, ls='--'); a.text(lim[1] - 3, lim[1] - 14, 'same under\npressure as usual', color=ORANGE, fontsize=7.5, ha='right')
    for p in el:
        if p['id'] in top: a.annotate(p['name'].split()[-1], (p['norm']['rae'], p['press']['rae']), xytext=(4, 3), textcoords='offset points', fontsize=7.5, color=INK)
    a.set_xlabel('Usual level: runs above expected per 100 balls (ordinary balls)'); a.set_ylabel('Under pressure: runs above expected per 100 balls'); a.set_title('Who rises above their own level? (above the line = better under pressure)', fontsize=9.5, color=INK, loc='left'); a.legend(frameon=False, fontsize=8, loc='upper left'); fig.tight_layout(); return save(fig, 'fig4_pressure_vs_usual.png')
# ---------- fig 5: nominal vs effective weights
def fig_eff():
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for a, m, t in ((ax[0], bm, 'Batting'), (ax[1], wm, 'Bowling')):
        k = m['metrics']['keys']; nom = np.array(m['metrics']['weights']) * 100; eff = np.array(m['effShare']) * 100; y = np.arange(len(k))[::-1]
        a.barh(y + .19, nom, height=.34, color=ORANGE, label='Nominal weight'); a.barh(y - .19, eff, height=.34, color=BLUE, label='Share of the final score it actually explains')
        a.set_yticks(y); a.set_yticklabels(k, fontsize=8); a.set_title(t, fontsize=10, color=INK, loc='left'); a.set_xlabel('% of composite'); a.grid(axis='y', visible=False)
    ax[0].legend(frameon=False, fontsize=7.5, loc='lower right'); fig.tight_layout(); return save(fig, 'fig5_nominal_vs_effective.png')
# ---------- fig 6: win-probability calibration
def fig_cal():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for a, inn in zip(ax, ('1', '2')):
        c = bm['cal'][inn]; a.plot([0, 1], [0, 1], color=GRID, lw=1.2); a.plot([r[0] for r in c], [r[1] for r in c], 'o-', color=BLUE, ms=4.5, lw=1.6, mec=SURF); a.set_title(f"Innings {inn} (out-of-fold)", fontsize=9.5, color=INK, loc='left'); a.set_xlabel('Predicted win probability'); a.set_ylabel('Actual win rate')
    fig.tight_layout(); return save(fig, 'fig6_win_probability_calibration.png')
F = {'tiers': fig_tiers(), 'rb': fig_rank(bt, 'fig2_rank_batters.png', 'Batters: how firm is each rank?'), 'rw': fig_rank(wt, 'fig3_rank_bowlers.png', 'Bowlers: how firm is each rank?'), 'sc': fig_scatter(), 'eff': fig_eff(), 'cal': fig_cal()}

# ---------- tables
f1 = lambda x, d=1: ('–' if x is None else f'{x:.{d}f}')
def bat_rows(n=10):
    return [[p['rank'], p['name'], p['grp'].split(' (')[0], p['pn'], p['pinn'], f1(p['press']['sr']), f1(p['press']['esr']), f1(p['press']['rae']), f1(p['norm']['rae']), f1(p['score']), f"{p['top10p']:.0%}", f"{p['rlo']}–{p['rhi']}", 'low sample' if p['lowconf'] else ''] for p in bt[:n]]
def bowl_rows(n=10):
    return [[p['rank'], p['name'], p['pn'], p['pinn'], f1(p['press']['econ'], 2), f1(p['press']['eecon'], 2), f1(p['norm']['econ'], 2), p['press']['wkts'], f1(p['score']), f"{p['top10p']:.0%}", f"{p['rlo']}–{p['rhi']}", 'low sample' if p['lowconf'] else ''] for p in wt[:n]]
BH = ['#', 'Batter', 'Slot', 'Pressure balls', 'Pressure inns', 'SR (pressure)', 'Expected SR', 'Runs above exp /100 (pressure)', '… (ordinary)', 'Clutch Meter', 'P(top-10)', 'Rank range', 'Flag']
WH = ['#', 'Bowler', 'Pressure balls', 'Pressure spells', 'Econ (pressure)', 'Expected econ', 'Econ (ordinary)', 'Wkts', 'Clutch Meter', 'P(top-10)', 'Rank range', 'Flag']
def metric_rows(m, why):
    k = m['metrics']; return [[n, nm, f"{w * 100:g}%", blk, f"{e * 100:.1f}%", f"{rp:.2f}", f"{vv:.2f}", why[n]] for n, nm, w, blk, e, rp, vv in zip(k['keys'], k['names'], k['weights'], k['block'], m['effShare'], k['repeat'], k['validity'])]
WHY_B = {'imp': 'Most repeatable signal in the data; the core of "performs under pressure".', 'wpa_cc': 'Brief: "how close the contest is". Win probability itself carries little repeatable signal.', 'big': 'Tests "when it matters" directly (knock-outs, top-4 opposition).', 'relief': 'Credit for explosive starts that stop pressure building; user-requested.',
         'delta': 'Brief: compare with usual levels. A difference of two noisy numbers, so kept below Level.', 'hit': 'Brief: "consistently". Counted per innings, so one big score cannot carry it.', 'floor': 'Penalises the bad innings (floor).', 'finish': 'Cricket test of delivering when the match is on the line.', 'protect': 'User-requested tail protection; gated on runs above expected.'}
WHY_W = {'runs': 'Only bowling signal that repeats well; largest weight.', 'win_cl': 'Classic test: defending the last 2 overs and close games.', 'qwk': 'Wickets matter but raw wickets are noise; weighted by batter quality and shrunk.', 'creation': 'Early wickets / dots that create pressure; user-requested.', 'delta': 'Brief: compare with usual levels.',
         'hit': 'Per-spell consistency.', 'floor': 'Penalises the bad spell.', 'bounce': 'Resilience: ball after conceding a boundary.', 'closing': 'Clean-up of the tail; user-requested analogue of tail protection.'}
def sens_rows(m):
    nm = lambda k: (k.replace('Role-standardised (score centred within batting slot / bowling role)', 'Position-standardised (score centred within batting position)' if m['kind'] == 'bat' else 'Standardised within groups of bowlers by share of overs bowled at the death') if 'Role-standardised' in k else k)
    return [[nm(k), f"{v['o10']} / 10", v['n']] for k, v in m['sens'].items()]
def audit_rows(m): return [[k, v['eligible'], f"{v['mean_z']:+.2f}", v['in_top10'], v['in_top30']] for k, v in m['audit'].items()]
bn, wn = [p['name'] for p in bt[:10]], [p['name'] for p in wt[:10]]
pb, pw = bm['pred'], wm['pred']; eb_, ew_ = bm['pred'], wm['pred']

# ---------- content blocks
blocks = []
H = lambda t, l=1: blocks.append(('h', l, t)); P = lambda t: blocks.append(('p', t)); BL = lambda xs: blocks.append(('b', xs)); TB = lambda h, r, cap=None: blocks.append(('t', h, r, cap)); IM = lambda f, cap: blocks.append(('i', f, cap))
H('SupeRR Selector Season 2 – Round 1: Most reliable players under pressure'); P('Methodology report (v4). All numbers below are produced by the code in pressure_model_v4/ from the supplied CSVs only; nothing relies on reputation.')
H('1. Summary of results', 2)
P('Final top 10 batters (rank order): ' + '; '.join(f'{i + 1}. {n}' for i, n in enumerate(bn)) + '.')
P('Final top 10 bowlers (rank order): ' + '; '.join(f'{i + 1}. {n}' for i, n in enumerate(wn)) + '.')
P('How the ranking works: every player gets a Clutch Meter (50 = the average eligible player, each 10 points = one standard deviation). It is 50 + 10 × the player’s average standardised score over 400 stress draws, in which I (a) resample the matches, (b) randomly perturb every metric weight by about ±35% and (c) switch at random between the strict and broad pressure definitions. Players are ranked strictly by Clutch Meter. The “P(top-10)” and rank range columns are confidence measures: they show how often, and how high, a player finishes across the 400 draws, and are not used to order the list.')
P('Honest headline: the bowling ranking is reasonably firm at the top (Bumrah is top-10 in ' + f"{wt[0]['top10p']:.0%}" + ' of draws; held-out validity ' + f"{pw['composite (final weights)']:.2f}" + '), while the batting ranking is only firm for the top handful (best P(top-10) ' + f"{bt[0]['top10p']:.0%}" + '; held-out validity ' + f"{pb['composite (final weights)']:.2f}" + '). Positions below the top 3–5 in each list are statistically close and should be read as a tier, not a strict order.')
H('2. Data preparation', 2)
BL([f"Scope: {bm['matches']} IPL and T20I matches from 2022 to {bm['dataEnd'][:10]}; {bm['rows']:,} faced deliveries (batting) and {wm['rows']:,} deliveries (bowling). Super-over innings and rain-affected (revised-target) matches are excluded because ball-level win probability is not comparable there.",
    f"Missing runs: the ball-by-ball file does not record the runs on byes, leg-byes or boundary wides (the 'extras1_runs' column is a type code, not a run count). The innings totals were therefore reconciled with the result text (chase won ⇒ chaser finishes above target; bat-first win by N runs ⇒ difference equals N) and the shortfall allocated to the extras deliveries. {bm['dq']['recon'].get('chase_total_topped_up', 0) + bm['dq']['recon'].get('first_total_topped_up', 0)} matches were adjusted this way and {bm['dq']['bad_matches']} with inconsistent data were dropped. This is a heuristic; it is the main data-quality caveat.",
    'Opponent strength: IPL end-of-season league table and ICC T20I team ratings (z-scored). These are season-end or year-end snapshots, so they contain some look-ahead; an Elo-style rolling rating would be cleaner.'])
H('3. Defining pressure', 2)
P('Pressure is defined once, as leverage: how much the win probability can swing on the next ball. A win-probability model (gradient boosting with monotone constraints, scored out-of-fold by match) is built from required rate, runs needed, wickets, balls left, venue-era par score, recent dot balls and a team-strength gap. Second-innings model: AUC ' + f"{bm['wp2']['auc']}" + ', Brier ' + f"{bm['wp2']['brier']} vs {bm['wp2']['base']}" + ' for a coin-flip baseline; first-innings AUC ' + f"{bm['wp1']['auc']}" + '. A second model predicts the expected absolute swing (leverage). A ball is then classed into a pressure tier:')
BL(['STRICT tier (headline, ' + f"{bm['strictShare']:.1%}" + ' of batting balls): top 10% of leverage (damped when the game is already decided), or a rule-based situation – last pair with 2+ overs left; rescue (4+ down inside 12 overs or 3+ down inside 6, as the underdog); early collapse (2+ wickets in 3 overs inside the first 6 overs); hard chase (target ≥ 1.15× venue par, first 8 overs, underdog).',
    'Bowling strict tier (' + f"{wm['strictShare']:.1%}" + ' of deliveries): death overs (16–20) with win chance 30–70%; the batting side scoring 26+ off the last 12 balls with the game live; a chase within 43–57% win chance; or top 10% of the bowling pressure index. The first-innings “knife-edge” rule was removed because first-innings win probability is near 50% early by construction, which made the powerplay look like a pressure phase.',
    'BROAD tier (' + f"{bm['broadShare']:.1%}" + ' of batting balls): the v3 definition (top 20% of a blended index plus extra situations), kept as one of the two definitions the final ranking averages over.'])
P('Does it distinguish pressure from ordinary play? Figure 1 and the table below compare the strict tier against ordinary balls for the same players.')
IM(F['tiers'], 'Figure 1. Strict pressure tier vs ordinary balls.')
TB(['Measure (batters)', 'Pressure', 'Ordinary'], [[k, bm['tiers']['Pressure (strict tier)'][k], bm['tiers']['Ordinary'][k]] for k in ('share', 'strike rate', 'dismissal % per ball', 'dot %', 'boundary %', 'sd of win-prob swing per ball (x100)', 'batting-side win %')])
TB(['Measure (bowlers)', 'Pressure', 'Ordinary'], [[k, wm['tiers']['Pressure (strict tier)'][k], wm['tiers']['Ordinary'][k]] for k in ('share', 'economy', 'wickets per over', 'dot %', 'boundary %', 'wides+no-balls %', 'fielding-side win %')])
BL(['For bowlers the tier is clearly harder: economy ' + f"{wm['tiers']['Pressure (strict tier)']['economy']} vs {wm['tiers']['Ordinary']['economy']}" + ', boundaries and extras up, dots down.',
    'For batters the tier is defined by consequence rather than by an obvious drop in output: the win-probability swing per ball is nearly 3× larger (' + f"{bm['tiers']['Pressure (strict tier)']['sd of win-prob swing per ball (x100)']} vs {bm['tiers']['Ordinary']['sd of win-prob swing per ball (x100)']}" + ') and the batting side wins only ' + f"{bm['tiers']['Pressure (strict tier)']['batting-side win %']}%" + ' of the time, but dismissal rate per ball is essentially unchanged and first-innings pressure strike rate is lower (112.6 vs 143.1) while chase pressure strike rate is higher (142.8 vs 136.0). I report this plainly: pressure for a batter is mostly about stakes, and the data show only modest behavioural change.'])
IM(F['cal'], 'Figure 2. Win-probability calibration by predicted decile (out-of-fold, grouped by match). The diagonal is perfect calibration.')
H('4. Measuring performance', 2)
P('Every delivery has an expected outcome from models that know the batter’s slot, the partner (quality, and whether that partner is a tail-ender), bowler quality, type and speed, venue, opposition strength, target difficulty and the match state (but never the player’s own identity). Player performance is the residual: actual minus expected, so an opener, a finisher and a death bowler are each compared with the average player in that situation. Batting impact adds the cost of getting out (the change in the batter’s expected future runs), so a six followed by a dismissal is not scored as a strike rate of 300.')
P('Each metric is an estimate with a known sampling variance (match-clustered, centred on the player’s own mean and never below the ball-level variance). DerSimonian–Laird empirical-Bayes shrinkage pulls it towards the population mean by exactly as much as its noise requires, and a metric with no repeatable between-player signal is shrunk to zero automatically. The composite is the weighted sum of the standardised, shrunk metrics.')
P('Weights were fixed in advance from the problem statement, not fitted: validity numbers in this data (Spearman ≈ 0–0.2 on 80–130 players, standard error ≈ 0.08–0.1) cannot distinguish one weighting from another. Blocks: Level 46% (how good under pressure), Pressure delta 18% (against the player’s own ordinary level – the brief’s “compared with usual performance”), Reliability 26% (“consistently”), pressure relief/creation 6% and tail protection/clean-up 4% (credit for openers who stop pressure building and for batters/bowlers who protect or finish the tail).')
H('Batting metrics', 3)
TB(['Key', 'Metric', 'Weight', 'Block', 'Effective share', 'Repeat (r)', 'Validity (r)', 'Why this weight'], metric_rows(bm, WHY_B))
H('Bowling metrics', 3)
TB(['Key', 'Metric', 'Weight', 'Block', 'Effective share', 'Repeat (r)', 'Validity (r)', 'Why this weight'], metric_rows(wm, WHY_W))
P('Repeat = Spearman correlation of the metric between two random halves of the matches; Validity = correlation of the metric in one half with held-out pressure performance in the other (strict tier, 20 random splits). “Effective share” is the part of the final score’s variance the metric actually explains.')
IM(F['eff'], 'Figure 3. Nominal weights vs the share of the final score each metric really drives.')
P('Important finding: the empirical-Bayes step finds no repeatable between-player signal in several of the metrics the brief asks for. In batting, about ' + f"{bm['effShare'][0] * 100:.0f}%" + ' of the final score comes from the impact metric; pressure delta, win-probability added, relief, downside, chase finishing and (almost) tail protection contribute close to nothing because they cannot be separated from luck with 8–40 pressure innings per player. In bowling, runs saved explains ' + f"{wm['effShare'][0] * 100:.0f}%" + ', the share of positive pressure spells ' + f"{wm['effShare'][5] * 100:.0f}%" + ' and pressure delta ' + f"{wm['effShare'][4] * 100:.0f}%" + '. The nominal weights above stay in the formula (they are what I would weight if the data allowed it), but the report does not claim that, for example, “clutch delta” is measured reliably.')
H('5. Fair comparisons', 2)
BL([f"Sample size: eligibility is {bm['minBalls']} strict-pressure balls and {bm['minInnings']} pressure innings for batters ({bm['eligible']} eligible) and {wm['minBalls']} deliveries and {wm['minInnings']} spells for bowlers ({wm['eligible']} eligible). Because the strict tier is small, these are lower than I would like; {bm['lowconf']} batters and {wm['lowconf']} bowlers are flagged ‘low sample’ and shrinkage is the main protection for them.",
    'Batting position: expectations are slot-aware; a position check (below) tests that the pooled batting list does not systematically favour one position.',
    'IPL vs T20I: both are included and the model has competition-level, opposition-strength and venue terms; the stress test includes IPL-only and T20I-only re-ranks. Opponent strength comes from IPL tables and ICC ratings, which are on different scales.',
    'Match context: matches are weighted by importance (knock-outs and must-win games count more) and by recency (3-year half-life); both are tested by removing them.',
    'Pool: every player in the data with enough pressure exposure is ranked. The brief does not require current activity, so no 2026 filter is applied (the dashboard shows an activity flag).'])
TB(['Batting position check', 'Eligible', 'Mean Clutch Meter (z)', 'In top 10', 'In top 30'], audit_rows(bm))
P('The batting list is balanced across positions (table above). For bowlers I ran the same check internally, grouping bowlers by the share of their deliveries bowled in overs 16–20; it found that bowlers who bowl many death overs score higher on average (7 of the top 10), because their pressure overs are by construction the highest-leverage ones. Centring the score within that grouping keeps 9 of the 10 (see the sensitivity table). The grouping is used only for this check and is not shown as a label anywhere.')
H('6. Evidence for the top 10', 2)
TB(BH, bat_rows()); IM(F['rb'], 'Figure 4. Rank ranges for the 15 highest-ranked batters.')
TB(WH, bowl_rows()); IM(F['rw'], 'Figure 5. Rank ranges for the 15 highest-ranked bowlers.')
IM(F['sc'], 'Figure 6. Each batter’s level under pressure against their own ordinary level.')
P('Reading Figure 6: players above the dashed line do better under pressure than usual; most good batters sit near the line, i.e. they are good players rather than players who lift specifically under pressure. This is why the score is described as “performance under pressure relative to the average player in that situation”, with the player’s own baseline shown alongside it.')
H('7. Reliability and limitations', 2)
TB(['Check', 'Batting', 'Bowling'], [['Held-out validity of composite, strict tier', f"{pb['composite (final weights)']:.3f}", f"{pw['composite (final weights)']:.3f}"], ['Held-out validity of composite, broad tier', f"{bm['predBroad']['composite (final weights)']:.3f}", f"{wm['predBroad']['composite (final weights)']:.3f}"],
    ['Split-half repeatability of composite (strict / broad)', f"{pb['composite split-half repeatability']:.3f} / {bm['predBroad']['composite split-half repeatability']:.3f}", f"{pw['composite split-half repeatability']:.3f} / {wm['predBroad']['composite split-half repeatability']:.3f}"],
    ['Chronological: 2022-23 → 2024-26 (strict)', bm['era']['2022-23 → 2024-26']['composite'], wm['era']['2022-23 → 2024-26']['composite']], ['Chronological: 2024-26 → 2022-23 (strict)', bm['era']['2024-26 → 2022-23']['composite'], wm['era']['2024-26 → 2022-23']['composite']],
    ['Mean P(top-10) of the final 10', f"{np.mean(bm['top10pFinal']):.0%}", f"{np.mean(wm['top10pFinal']):.0%}"], ['Players with P(top-10) ≥ 80%', sum(1 for p in bt if p['top10p'] >= .8), sum(1 for p in wt if p['top10p'] >= .8)]])
P('Sensitivity of the final top 10 (number of the final 10 recovered by each variant):')
TB(['Batting variant', 'Overlap', 'Eligible in variant'], sens_rows(bm)); TB(['Bowling variant', 'Overlap', 'Eligible in variant'], sens_rows(wm))
BL(['Batting is a weak-signal problem. Validity of 0.03–0.04 means the ordering mostly reflects luck beyond the top few; only the tier (strong middle-order/finisher/opener hitters) is dependable. Of the 10 ranked, the best has a P(top-10) of ' + f"{bt[0]['top10p']:.0%}" + ' and several are flagged low-sample.',
    'The result depends on the data window: re-ranking on T20I matches only recovers just ' + f"{bm['sens']['T20I / other matches only']['o10']} of 10 batters and {wm['sens']['T20I / other matches only']['o10']} of 10 bowlers" + ' (T20I samples are small, so this is partly noise), and IPL-only recovers ' + f"{bm['sens']['IPL matches only']['o10']} / {wm['sens']['IPL matches only']['o10']}" + '. The final list is effectively an IPL-weighted view.',
    'The pressure definition matters: the strict-only and broad-only top 10s each recover 7–8 batters and 9–10 bowlers; averaging both is the compromise.',
    'Run reconstruction, season-end team ratings and absence of pitch/dew information are known data limitations. Bye/leg-bye runs are allocated, not observed.',
    'Because the weights are fixed rather than fitted, they reflect my judgement of the brief; the effective-share chart shows how little of that judgement the data can currently confirm.'])
H('8. Reproducing the results', 2)
BL(['Environment: Python 3.12 with numpy, pandas, scikit-learn, scipy (see requirements.txt).', 'python m1_features.py - .. stage1.pkl  →  python m2_expect.py - stage1.pkl stage2.pkl ..  →  python b1_bowling_expect.py - stage2.pkl bstage.pkl ..',
    'python v4_rate.py - bat stage2.pkl .. batters_v4.json  and  python v4_rate.py - bowl bstage.pkl .. bowlers_v4.json   (BR=400 draws; seeds fixed)', 'python make_report.py   and   python build_dashboard_v4.py'])

# ---------- render
def to_md():
    L = []
    for b in blocks:
        if b[0] == 'h': L.append('#' * b[1] + ' ' + b[2] + '\n')
        elif b[0] == 'p': L.append(b[1] + '\n')
        elif b[0] == 'b': L.append('\n'.join('- ' + x for x in b[1]) + '\n')
        elif b[0] == 't': L.append('| ' + ' | '.join(map(str, b[1])) + ' |\n|' + '---|' * len(b[1]) + '\n' + '\n'.join('| ' + ' | '.join(map(str, r)) + ' |' for r in b[2]) + '\n')
        elif b[0] == 'i': L.append(f"![{b[2]}]({os.path.basename(b[1])})\n\n*{b[2]}*\n")
    return '\n'.join(L)
open(os.path.join(OUT, 'methodology_report.md'), 'w').write(to_md())
doc = Document(); doc.styles['Normal'].font.size = Pt(10); doc.styles['Normal'].font.name = 'Calibri'
for s_ in doc.sections: s_.left_margin = s_.right_margin = Inches(0.8); s_.top_margin = s_.bottom_margin = Inches(0.8)
for b in blocks:
    if b[0] == 'h': doc.add_heading(b[2], level=min(b[1], 3) if b[1] > 1 else 0)
    elif b[0] == 'p': doc.add_paragraph(b[1])
    elif b[0] == 'b':
        for x in b[1]: doc.add_paragraph(x, style='List Bullet')
    elif b[0] == 't':
        t = doc.add_table(rows=1, cols=len(b[1])); t.style = 'Light Grid Accent 1'
        for i, h in enumerate(b[1]): t.rows[0].cells[i].text = str(h)
        for r in b[2]:
            c = t.add_row().cells
            for i, v in enumerate(r): c[i].text = str(v)
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    for r in p.runs: r.font.size = Pt(7 if len(b[1]) > 6 else 9)
        doc.add_paragraph()
    elif b[0] == 'i': doc.add_picture(b[1], width=Inches(6.2)); doc.add_paragraph(b[2]).italic = True
doc.save(os.path.join(OUT, 'methodology_report.docx')); print('wrote', OUT)
