"""usage: python build_dashboard.py pressure_all.json pressure_active.json bowlers_all.json bowlers_active.json out.html"""
import sys, os
here = os.path.dirname(os.path.abspath(__file__)); rd = lambda f: open(f).read().replace('</', '<\\/')
html = open(os.path.join(here, 'app_template.html')).read()
for k, f in zip(('__BAT_ALL__', '__BAT_ACT__', '__BOWL_ALL__', '__BOWL_ACT__'), sys.argv[1:5]): html = html.replace(k, rd(f))
open(sys.argv[5], 'w').write(html); print('wrote', sys.argv[5], len(html) // 1024, 'KB')
