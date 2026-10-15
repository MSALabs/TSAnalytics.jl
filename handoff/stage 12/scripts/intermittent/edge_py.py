import numpy as np, warnings
from statsforecast.models import (CrostonClassic, CrostonOptimized, CrostonSBA, TSB, ADIDA, IMAPA)
CASES = {
 "all_zeros_n10":        np.zeros(10),
 "all_zeros_n1":         np.zeros(1),
 "one_nonzero_n10_mid":  np.array([0,0,0,5,0,0,0,0,0,0.]),
 "one_nonzero_n10_last": np.array([0,0,0,0,0,0,0,0,0,5.]),
 "one_nonzero_n10_first":np.array([5,0,0,0,0,0,0,0,0,0.]),
 "len1_nonzero":         np.array([3.]),
 "len2_0_3":             np.array([0,3.]),
 "len2_3_0":             np.array([3,0.]),
 "len2_3_4":             np.array([3,4.]),
 "two_nonzero_end_nz":   np.array([0,2,0,0,3.]),
 "two_nonzero_end_zero": np.array([0,2,0,3,0.]),
 "no_zeros_n6":          np.array([1,2,3,2,1,2.]),
}
def g(v):
    try: return f"{float(v):.12g}"
    except Exception: return str(v)
mods=[("CrostonClassic",lambda:CrostonClassic()),("CrostonOptimized",lambda:CrostonOptimized()),
      ("CrostonSBA",lambda:CrostonSBA()),("TSB",lambda:TSB(alpha_d=0.2,alpha_p=0.2)),
      ("ADIDA",lambda:ADIDA()),("IMAPA",lambda:IMAPA())]
print(f"{'case':<24}{'model':<18}{'h=1..3':<42}note")
for cn, y in CASES.items():
    for mn, mk in mods:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r=mk().forecast(y=y,h=3,fitted=False)
                v=" ".join(g(x) for x in r["mean"])
                note=""
            except Exception as e:
                v="RAISED"; note=f"{type(e).__name__}: {str(e)[:70]}"
        print(f"{cn:<24}{mn:<18}{v:<42}{note}")
    print()
