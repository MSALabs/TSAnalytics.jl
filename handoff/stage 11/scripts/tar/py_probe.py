import importlib, pkgutil, re, sys
import statsmodels, statsmodels.tsa as tsa
print("statsmodels", statsmodels.__version__)
mods = sorted(m.name for m in pkgutil.walk_packages(tsa.__path__, prefix="statsmodels.tsa."))
print("n tsa submodules:", len(mods))
pat = re.compile(r"setar|threshold|thresh|star|tar_|regime|transition", re.I)
hits = [m for m in mods if pat.search(m)]
print("name hits:", hits)
# search the installed source tree for the words
