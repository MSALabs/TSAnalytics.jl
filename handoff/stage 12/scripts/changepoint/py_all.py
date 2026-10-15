import numpy as np, ruptures as rpt, csv, os, time
D="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/changepoint"
def load(n):
    with open(os.path.join(D,n)) as f: return np.array([float(r[0]) for r in list(csv.reader(f))[1:]])
FX=["fx0_regen","fxA_var","fxB_weak","fxC_null","fxD_edge","fxF_offgrid","fxG_many","fxH_spike"]
for fx in FX:
    y=load(fx+".csv"); n=len(y)
    print(f"\n################ {fx}  n={n}")
    for pen in (10,20,50):
        for jump,ms in ((5,2),(1,1),(1,2)):
            b=rpt.Pelt(model="l2",min_size=ms,jump=jump).fit(y).predict(pen=pen)
            print(f"  Pelt(l2) pen={pen:<3} jump={jump} min_size={ms} : {b}")
    # changepoint-equivalent penalties
    for nm,pen in (("BIC=2log n",2*np.log(n)),("MBIC-pen=3log n",3*np.log(n))):
        b=rpt.Pelt(model="l2",min_size=1,jump=1).fit(y).predict(pen=pen)
        print(f"  Pelt(l2) {nm}={pen:.6f} jump=1 ms=1 : {b}")
    # Binseg
    for q in (2,5):
        try:
            b=rpt.Binseg(model="l2",min_size=1,jump=1).fit(y).predict(n_bkps=q)
            print(f"  Binseg(l2) n_bkps={q} jump=1 ms=1 : {b}")
        except Exception as e: print(f"  Binseg n_bkps={q}: ERR {type(e).__name__}")
    b=rpt.Binseg(model="l2").fit(y).predict(pen=2*np.log(n))
    print(f"  Binseg(l2) DEFAULTS pen=2log n : {b}")
    # Dynp exact DP
    for q in (2,):
        b=rpt.Dynp(model="l2",min_size=1,jump=1).fit(y).predict(n_bkps=q)
        print(f"  Dynp(l2) n_bkps={q} jump=1 ms=1 : {b}")
    # normal (per-segment variance)
    b=rpt.Pelt(model="normal",min_size=2,jump=1).fit(y).predict(pen=3*np.log(n))
    print(f"  Pelt(normal) pen=3log n jump=1 ms=2 : {b}")
    # rbf / l1 / rank
    for mdl in ("rbf","l1","rank","ar"):
        try:
            b=rpt.Pelt(model=mdl,min_size=2,jump=1).fit(y).predict(pen=3*np.log(n))
            print(f"  Pelt({mdl}) pen=3log n jump=1 : {b}")
        except Exception as e: print(f"  Pelt({mdl}): ERR {type(e).__name__}: {e}")
    # Window / BottomUp / KernelCPD
    try:
        b=rpt.Window(width=40,model="l2",min_size=2,jump=1).fit(y).predict(pen=3*np.log(n))
        print(f"  Window(w=40,l2) pen=3log n : {b}")
    except Exception as e: print(f"  Window: ERR {e}")
    try:
        b=rpt.BottomUp(model="l2",min_size=2,jump=1).fit(y).predict(pen=3*np.log(n))
        print(f"  BottomUp(l2) pen=3log n : {b}")
    except Exception as e: print(f"  BottomUp: ERR {e}")
    try:
        b=rpt.KernelCPD(kernel="linear",min_size=2).fit(y).predict(pen=3*np.log(n))
        print(f"  KernelCPD(linear) pen=3log n : {b}")
    except Exception as e: print(f"  KernelCPD: ERR {e}")
