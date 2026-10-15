import numpy as np, csv, os
BASE=os.path.dirname(os.path.abspath(__file__))
def write(name, y, note):
    p=os.path.join(BASE,name)
    with open(p,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['y'])
        for v in y: w.writerow([int(v)])
    nz=int((y>0).sum())
    print(f"{name:<26} n={len(y):>4} nonzero={nz:>4} sum={int(y.sum()):>5} "
          f"ADI={len(y)/nz if nz else float('nan'):.6f}  # {note}")
    return p

# F1: existing (re-derive to confirm provenance claim: Poisson(0.4), n=60, seed 7)
for gen,label in [(lambda: np.random.default_rng(7).poisson(0.4,60),'np.default_rng(7)'),
                  (lambda: (np.random.seed(7), np.random.poisson(0.4,60))[1],'np.random.seed(7) legacy')]:
    y=gen(); print(label, 'n60 p0.4 ->', 'nonzero',int((y>0).sum()),'sum',int(y.sum()))

# F2 sparse, F3 dense  (seed recorded; CSV is the fixture of record)
rng2=np.random.default_rng(101); y2=rng2.poisson(0.1,120)
rng3=np.random.default_rng(202); y3=rng3.poisson(1.5,120)
write('fx_sparse_p010_n120.csv', y2, 'np.random.default_rng(101).poisson(0.1,120)')
write('fx_dense_p150_n120.csv',  y3, 'np.random.default_rng(202).poisson(1.5,120)')
