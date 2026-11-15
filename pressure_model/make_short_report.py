"""Builds the short (5-6 page) plain-language report as PDF via headless Chrome.  usage: python make_short_report.py"""
import json, os, base64, subprocess, re
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, 'report_v4')
B = json.load(open(os.path.join(H, 'batters_v4.json'))); W = json.load(open(os.path.join(H, 'bowlers_v4.json'))); bm, wm = B['meta'], W['meta']
bt = sorted([p for p in B['players'] if p['elig']], key=lambda p: p['rank']); wt = sorted([p for p in W['players'] if p['elig']], key=lambda p: p['rank'])
img = lambda f: 'data:image/png;base64,' + base64.b64encode(open(os.path.join(R, f), 'rb').read()).decode()
f1 = lambda x, d=1: f'{x:.{d}f}'
pos = lambda g: g.split(' (')[0]
def tbl(head, rows, cls=''): return f'<table class="{cls}"><thead><tr>' + ''.join(f'<th>{h}</th>' for h in head) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows) + '</tbody></table>'
top_b = tbl(['#', 'Batter', 'Position', 'Clutch Meter', 'In top 10 in'], [[p['rank'], p['name'], pos(p['grp']), f1(p['score']), f"{p['top10p']:.0%}"] for p in bt[:10]])
top_w = tbl(['#', 'Bowler', 'Clutch Meter', 'In top 10 in'], [[p['rank'], p['name'], f1(p['score']), f"{p['top10p']:.0%}"] for p in wt[:10]])
ev_b = tbl(['#', 'Batter', 'Pressure balls', 'Pressure innings', 'Strike rate (expected)', 'Runs above expected per 100 balls: pressure | usual'], [[p['rank'], p['name'], p['pn'], p['pinn'], f"{f1(p['press']['sr'])} ({f1(p['press']['esr'])})", f"{p['press']['rae']:+.0f} | {p['norm']['rae']:+.0f}"] for p in bt[:10]], 'sm')
ev_w = tbl(['#', 'Bowler', 'Pressure balls', 'Pressure spells', 'Economy (expected)', 'Usual economy'], [[p['rank'], p['name'], p['pn'], p['pinn'], f"{f1(p['press']['econ'], 2)} ({f1(p['press']['eecon'], 2)})", f1(p['norm']['econ'], 2)] for p in wt[:10]], 'sm')
tb, tw = bm['tiers'], wm['tiers']
vs = tbl(['', 'Batters: pressure', 'ordinary', 'Bowlers: pressure', 'ordinary'], [['Share of all balls', f"{tb['Pressure (strict tier)']['share']:.0%}", f"{tb['Ordinary']['share']:.0%}", f"{tw['Pressure (strict tier)']['share']:.0%}", f"{tw['Ordinary']['share']:.0%}"],
    ['Strike rate / economy', tb['Pressure (strict tier)']['strike rate'], tb['Ordinary']['strike rate'], tw['Pressure (strict tier)']['economy'], tw['Ordinary']['economy']], ['Boundary %', tb['Pressure (strict tier)']['boundary %'], tb['Ordinary']['boundary %'], tw['Pressure (strict tier)']['boundary %'], tw['Ordinary']['boundary %']],
    ['Dot ball %', tb['Pressure (strict tier)']['dot %'], tb['Ordinary']['dot %'], tw['Pressure (strict tier)']['dot %'], tw['Ordinary']['dot %']], ['Win-probability swing per ball (x100)', tb['Pressure (strict tier)']['sd of win-prob swing per ball (x100)'], tb['Ordinary']['sd of win-prob swing per ball (x100)'], '', ''],
    ['Fielding side wins %', '', '', tw['Pressure (strict tier)']['fielding-side win %'], tw['Ordinary']['fielding-side win %']]])
sb, sw = bm['sens'], wm['sens']; k_t20 = 'T20I / other matches only'; k_ipl = 'IPL matches only'
rel = tbl(['Check', 'Batting', 'Bowling'], [['Held-out validity of the final score (0 = no signal)', f"{bm['pred']['composite (final weights)']:.2f}", f"{wm['pred']['composite (final weights)']:.2f}"], ['Same split-half repeatability', f"{bm['pred']['composite split-half repeatability']:.2f}", f"{wm['pred']['composite split-half repeatability']:.2f}"],
    ['Best player\'s chance of being top 10', f"{bt[0]['top10p']:.0%}", f"{wt[0]['top10p']:.0%}"], ['Players with 80%+ chance of top 10', sum(p['top10p'] >= .8 for p in bt), sum(p['top10p'] >= .8 for p in wt)],
    ['Same top 10 if strict pressure only (of 10)', sb['Strict tier only (~17% of balls)']['o10'], sw['Strict tier only (~17% of balls)']['o10']], ['Same top 10 if broad pressure only (of 10)', sb['Broad tier only (v3 definition, ~33% of balls)']['o10'], sw['Broad tier only (v3 definition, ~33% of balls)']['o10']],
    ['Same top 10 using IPL matches only (of 10)', sb[k_ipl]['o10'], sw[k_ipl]['o10']], ['Same top 10 using internationals only (of 10)', sb[k_t20]['o10'], sw[k_t20]['o10']], ['Players flagged "few pressure innings" (fewer than 10)', bm['lowconf'], wm['lowconf']]])
eff = lambda m, k: m['effShare'][m['metrics']['keys'].index(k)] * 100
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
@page{{size:A4;margin:12mm 14mm 12mm 14mm}}*{{box-sizing:border-box}}body{{font:9.7pt/1.38 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:#1c2b3a;margin:0}}
h1{{font-size:21pt;margin:0 0 2pt;color:#1e3a5f}}h2{{font-size:14pt;color:#1e3a5f;border-bottom:2px solid #4a90d9;padding-bottom:3pt;margin:0 0 8pt}}h3{{font-size:10.5pt;margin:7pt 0 2pt;color:#2a78d6}}p{{margin:4pt 0}}ul{{margin:3pt 0 6pt 16pt;padding:0}}li{{margin:2.5pt 0}}
.page{{page-break-after:always}}.page:last-child{{page-break-after:auto}}.sub{{color:#5b6b7b;font-size:10pt;margin-bottom:10pt}}
table{{border-collapse:collapse;width:100%;font-size:9pt;margin:3pt 0 6pt}}th{{background:#eaf1fa;text-align:left;padding:4pt 6pt;font-weight:600;border-bottom:1px solid #c9d6e6}}td{{padding:2.8pt 6pt;border-bottom:1px solid #e3e9f0}}table.sm{{font-size:8.7pt}}table.sm td,table.sm th{{padding:3pt 5pt}}
.cols{{display:flex;gap:14pt}}.cols>div{{flex:1}}.box{{background:#f2f6fb;border-left:4px solid #4a90d9;padding:7pt 10pt;margin:8pt 0;border-radius:0 6pt 6pt 0}}.warn{{background:#fff6f1;border-left-color:#eb6834}}img{{max-width:100%}}.c2 img{{width:100%}}
.steps{{display:flex;gap:8pt;margin:8pt 0}}.steps div{{flex:1;background:#f2f6fb;border-radius:8pt;padding:8pt;font-size:9.6pt}}.steps b{{display:block;color:#2a78d6;margin-bottom:2pt}}.foot{{color:#5b6b7b;font-size:8.5pt;margin-top:8pt}}
</style></head><body>
<div class="page"><h1>Who delivers under pressure?</h1><div class="sub">SupeRR Selector Season 2 · Round 1 · by Aarush Singh · aarushsingh198@gmail.com · short report (full method in the long report)</div>
<h2>1. The answer</h2><div class="cols"><div><h3>Top 10 batters</h3>{top_b}</div><div><h3>Top 10 bowlers</h3>{top_w}</div></div>
<p class="foot">Clutch Meter: 50 = average eligible player, 10 points = one standard deviation. Players are ranked strictly by Clutch Meter. “In top 10 in” is a confidence measure only: the share of 400 stress runs in which the player finishes in the top 10 (page 4).</p>
<div class="box"><b>How firm is this?</b><ul><li><b>Bowling is fairly firm.</b> Bumrah is in the top 10 in {wt[0]['top10p']:.0%} of runs; the final score predicts held-out pressure results (validity {wm['pred']['composite (final weights)']:.2f}).</li>
<li><b>Batting is firm only for the first few.</b> The best chance is {bt[0]['top10p']:.0%} and validity is only {bm['pred']['composite (final weights)']:.2f}. Positions 4–10 are close; read them as a group, not a strict order.</li></ul></div>
<h2 style="margin-top:14pt">2. What we did, in four steps</h2><div class="steps"><div><b>1. Define pressure</b>Moments when one ball can swing the match (win-probability model).</div><div><b>2. Set a fair baseline</b>What an average player would do in the same situation.</div><div><b>3. Score players</b>Actual minus expected, plus a player’s own usual level and consistency.</div><div><b>4. Stress-test</b>Re-rank 400 times with different matches, weights and pressure definitions.</div></div>
<p>Data: {bm['matches']} IPL and T20I matches (2022 to {bm['dataEnd'][:7]}), {bm['rows']:,} balls faced. Super overs and rain-affected matches are left out.</p></div>

<div class="page"><h2>3. What counts as pressure</h2><p>A ball is “pressure” when the match result can swing a lot on it. We built a win-probability model (chase accuracy AUC {bm['wp2']['auc']}, first innings {bm['wp1']['auc']}) and measured how much each ball can move it.</p>
<h3>Batting pressure (about {bm['strictShare']:.0%} of balls)</h3><ul><li>The top 10% of balls by how much the result can swing, or</li><li>Last pair batting with 2+ overs left; a collapse (4 wickets down inside 12 overs, or 3 down inside 6) while behind; two quick early wickets; a very hard chase (target 15%+ above par) early on.</li></ul>
<h3>Bowling pressure (about {wm['strictShare']:.0%} of balls)</h3><ul><li>Death overs with the match still open (30–70% win chance); being hit for 26+ in 12 balls with the game live; a tight chase (43–57% win chance); or the top 10% of high-swing balls.</li></ul>
<p>A second, broader definition ({bm['broadShare']:.0%} of balls) is also used. The final list averages both, so it does not depend on one choice.</p>
<h3>Does it separate pressure from ordinary play?</h3>{vs}
<div class="box"><ul><li><b>Bowlers:</b> clearly harder under pressure (economy {tw['Pressure (strict tier)']['economy']} vs {tw['Ordinary']['economy']}, more boundaries and extras).</li><li><b>Batters:</b> pressure is about <b>stakes</b>. The result swings almost 3 times more per ball and the batting side wins only {tb['Pressure (strict tier)']['batting-side win %']}% of the time, but dismissal rates barely change. We say this plainly instead of overstating it.</li></ul></div>
<img src="{img('fig1_pressure_vs_ordinary.png')}" style="width:82%;display:block;margin:2pt auto"></div>

<div class="page"><h2>4. How players are scored</h2><p>Every ball has an <b>expected</b> result for an average player in that spot, using batting position, partner, bowler (or batter) quality, venue, opposition strength, target and match state. A player’s performance is <b>actual minus expected</b>, so openers, finishers and death bowlers are each judged against their own role.</p>
<table><thead><tr><th>Part of the score</th><th>Weight</th><th>Batters</th><th>Bowlers</th></tr></thead><tbody>
<tr><td><b>Level</b></td><td>46%</td><td>Runs above expected under pressure (including the cost of getting out), win-probability added in chases, impact against top sides and in knock-outs</td><td>Runs saved vs expected, win impact defending the last 2 overs and close games, quality-weighted wickets</td></tr>
<tr><td><b>Pressure delta</b></td><td>18%</td><td colspan="2">Pressure performance minus the player’s own usual level (the brief’s “compared with usual”)</td></tr>
<tr><td><b>Reliability</b></td><td>26%</td><td>Share of pressure innings that were positive, size of bad innings, finishing chases not out</td><td>Share of positive spells, size of bad spells, bounce-back after a boundary</td></tr>
<tr><td><b>Relief / creation</b></td><td>6%</td><td>Explosive starts in overs 1–6 that stop pressure building</td><td>Early wickets and dots that create pressure</td></tr>
<tr><td><b>Tail</b></td><td>4%</td><td>Protecting the strike with a tail-ender at the other end</td><td>Cleaning up the tail</td></tr></tbody></table>
<h3>Small samples are pulled towards the average</h3><p>A player with few pressure innings can look great or terrible by luck. Every metric is “shrunk” towards the average by exactly as much as its noise requires. A metric with no repeatable signal is shrunk to zero automatically.</p>
<div class="box warn"><b>Important finding.</b> Only some metrics carry real, repeatable signal. In batting about <b>{eff(bm, 'imp'):.0f}%</b> of the final score comes from one metric (runs above expected under pressure); delta, relief, reliability and tail barely move the result. In bowling, runs saved explains about <b>{eff(wm, 'runs'):.0f}%</b>, share of positive spells {eff(wm, 'hit'):.0f}% and pressure delta {eff(wm, 'delta'):.0f}%. The weights above are what we <i>chose</i>; the chart shows what the data lets them do.</div>
<img src="{img('fig5_nominal_vs_effective.png')}" style="width:88%;display:block;margin:2pt auto"></div>

<div class="page"><h2>5. Fair comparisons</h2><ul>
<li><b>Same baseline for everyone:</b> a player is only compared with an average player in the same spot, so no role is automatically favoured.</li>
<li><b>Minimum sample:</b> batters need 60+ pressure balls and 8+ pressure innings; bowlers 80+ balls and 8+ spells. Players with fewer than 10 pressure innings carry a “few pressure innings” flag in the dashboard.</li>
<li><b>Lucky streaks are discounted:</b> balls inside one innings are not independent, so the model counts innings, not just balls.</li>
<li><b>IPL and T20I together:</b> both are used, with allowance for league, opposition strength (IPL table, ICC ratings) and venue. The dashboard can re-score any year window and either league separately.</li>
<li><b>Match importance and recency:</b> knock-out and must-win games count more; recent matches count a little more (3-year half-life). Both are tested by removing them.</li>
<li><b>Everyone is ranked:</b> no “active in 2026” filter in the submitted list, because the brief does not ask for one.</li></ul>
<h2 style="margin-top:12pt">6. How the ranking is built</h2><ul><li><b>Clutch Meter</b> = 50 + 10 × the player’s average standardised score over <b>400 stress runs</b>. Each run changes three things: which matches are included (resampling), how much each metric weighs (±35%) and which pressure definition is used (strict or broad).</li>
<li>Players are <b>ranked strictly by Clutch Meter</b>. Averaging over the runs means the list does not depend on any single modelling choice of ours.</li>
<li><b>“In top 10 in” and the rank range</b> are shown beside every rank as confidence: a high chance and a short bar mean a firm rank.</li></ul>
<div class="c2 cols"><div><img src="{img('fig2_rank_batters.png')}"></div><div><img src="{img('fig3_rank_bowlers.png')}"></div></div>
<p class="foot">Bars show where each player lands across the 400 runs (5th to 95th percentile); dots are medians. Long bars mean an uncertain rank.</p></div>

<div class="page"><h2>7. Evidence for the top 10</h2><h3>Batters (strict pressure situations)</h3>{ev_b}<h3>Bowlers (strict pressure situations)</h3>{ev_w}
<ul><li><b>Batters:</b> strike rate under pressure is compared with what an average player would score in the same situations (in brackets). Most have a large positive gap.</li>
<li><b>Bowlers:</b> they concede fewer runs under pressure than an average bowler would in the same situations; many are also good in ordinary play, so “level” matters more than a big pressure-only jump.</li></ul></div>

<div class="page"><h2>8. How reliable is this?</h2>{rel}
<h3>Plain-language limits</h3><ul>
<li><b>Batting signal is weak.</b> With only 8–40 pressure innings per player, much of the ordering beyond the top few is luck. That is why we report “chance of being in the top 10” beside every rank.</li>
<li><b>The pressure definition matters.</b> Using only strict or only broad pressure changes the batting top 10 by 2–3 names, bowling by 0–1.</li>
<li><b>The result leans on IPL data.</b> Internationals-only samples are small, so the international top 10 is quite different. Treat league views as supporting detail.</li>
<li><b>Death-over bowlers score a little higher on average</b> (7 of the top 10 bowl many death overs), because their pressure overs are by construction the highest-stakes. Adjusting for this keeps 9 of the 10.</li>
<li><b>Missing runs are estimated.</b> The data does not record bye and leg-bye runs, so about {bm['dq']['recon'].get('chase_total_topped_up', 0) + bm['dq']['recon'].get('first_total_topped_up', 0)} match totals were reconciled with the result text and the gap was allocated to extras balls.</li>
<li><b>Team strength looks slightly ahead</b> (season-end tables and year-end ratings), not strictly “as known before the match”.</li></ul>
<h3>Where to look</h3><ul><li><b>Dashboard</b> (dashboard_v4.html): filter by position, pace/spin, year window and league, sort any column, click a row for the breakdown.</li><li><b>Long report</b> (methodology_report.docx): full method, every metric, sensitivity tables.</li><li><b>Code</b> (pressure_model_v4/): run in the order given in README.txt; seeds are fixed.</li></ul></div>
</body></html>'''
out_html = os.path.join(R, 'short_report.html'); open(out_html, 'w').write(HTML)
C = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'; pdf = os.path.join(R, 'short_report.pdf')
subprocess.run([C, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={pdf}', 'file://' + out_html], stderr=subprocess.DEVNULL, check=True)
n = len(re.findall(rb'/Type\s*/Page[^s]', open(pdf, 'rb').read())); print('pages:', n, pdf)
