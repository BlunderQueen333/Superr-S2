Pressure model v4 - rebuild order (Python 3.12; pip install -r requirements.txt). DATA = folder with the original CSVs (one level up). Seeds are fixed.
  python m1_features.py - .. stage1.pkl                      # clean totals, win probability + leverage, team strength
  python m2_expect.py   - stage1.pkl stage2.pkl ..           # STRICT + BROAD pressure tiers, slot/partner/bowler-aware expected outcomes, strike-protection data
  python b1_bowling_expect.py - stage2.pkl bstage.pkl ..     # bowling expected outcomes + strict/broad bowling pressure tiers
  BR=400 python v4_rate.py - bat  stage2.pkl .. batters_v4.json     # 9 batting metrics, shrinkage, validation, stress test, audit
  BR=400 python v4_rate.py - bowl bstage.pkl .. bowlers_v4.json     # 9 bowling metrics, same machinery
  python make_report.py                                      # report_v4/methodology_report.docx + .md + charts
  python build_dashboard_v4.py                               # ../dashboard_v4.html
Files kept from v3 for reference: m4_rate.py, b2_rate.py, build_dashboard.py, app_template.html (old pillar-based scoring; not used by v4).
Honest summary: held-out validity of the composite - batting ~0.03-0.04, bowling ~0.14-0.17. Beyond the top few, ranks are close to a tier.
