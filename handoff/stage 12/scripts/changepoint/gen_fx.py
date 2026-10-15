import numpy as np, os
D = "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/changepoint"

def save(name, y):
    p = os.path.join(D, name)
    with open(p, "w") as f:
        f.write("y\n")
        for v in y:
            f.write("%.17g\n" % v)
    print("%-16s n=%-5d sum=%.10f  sd=%.10f" % (name, len(y), y.sum(), y.std(ddof=1)))

# fx0 reference: regenerate the starter to confirm the recipe
r = np.random.default_rng(20261015)
y = r.normal(0,1,300); y[100:200] += 5.0
save("fx0_regen.csv", y)

# (a) variance change only: sd 1 -> 3 -> 1, mean 0 throughout
r = np.random.default_rng(1001)
y = np.concatenate([r.normal(0,1,100), r.normal(0,3,100), r.normal(0,1,100)])
save("fxA_var.csv", y)

# (b) mean change of 0.5 sd at t=150 only
r = np.random.default_rng(1002)
y = r.normal(0,1,300); y[150:] += 0.5
save("fxB_weak.csv", y)

# (c) no change at all
r = np.random.default_rng(1003)
y = r.normal(0,1,300)
save("fxC_null.csv", y)

# (d) change at very start (t=3) and very end (t=297), n=300
r = np.random.default_rng(1004)
y = r.normal(0,1,300); y[3:297] += 4.0
save("fxD_edge.csv", y)

# (e) n=2000, 4 changes, for timing
r = np.random.default_rng(1005)
y = r.normal(0,1,2000)
for s,e,d in [(400,800,3.0),(800,1200,-2.0),(1200,1600,2.5)]:
    y[s:e] += d
save("fxE_n2000.csv", y)

# (f) off-grid changepoint at t=97 and t=203 (NOT multiples of 5) to expose jump
r = np.random.default_rng(1006)
y = r.normal(0,1,300); y[97:203] += 5.0
save("fxF_offgrid.csv", y)

# (g) many small segments: MBIC vs BIC should disagree on count
r = np.random.default_rng(1007)
y = r.normal(0,1,300)
for k in range(1,10):               # 9 changes, alternating +/-1.2, 30-long segments
    y[30*k:30*(k+1)] += 1.2*((-1)**k)
save("fxG_many.csv", y)

# (h) short-segment fixture: a 1-point and a 2-point spike, to test minseglen
r = np.random.default_rng(1008)
y = r.normal(0,1,200)
y[50] += 12.0          # single-point spike -> needs minseglen 1
y[120:122] += 12.0     # two-point spike
save("fxH_spike.csv", y)
