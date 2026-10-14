import numpy as np, json
# ---- FIXTURE: simulate from a known DFM, 1 factor, N=5, T=200 ----
def sim(seed, T, N, Lam, A, sig2, burn=200):
    rng = np.random.default_rng(seed)
    r = Lam.shape[1]
    f = np.zeros((T+burn, r))
    eta = rng.standard_normal((T+burn, r))          # Var(eta)=I
    for t in range(1, T+burn):
        f[t] = A @ f[t-1] + eta[t]
    f = f[burn:]
    eps = rng.standard_normal((T, N)) * np.sqrt(sig2)
    y = f @ Lam.T + eps
    return y, f

# fixture A: r=1
LamA = np.array([[1.0],[0.8],[-0.6],[0.5],[0.9]])
AA   = np.array([[0.7]])
sig2A= np.array([0.30,0.50,0.40,0.60,0.35])
yA, fA = sim(20260214, 200, 5, LamA, AA, sig2A)

# fixture B: r=2, N=6, T=250
LamB = np.array([[1.0, 0.0],[0.8, 0.4],[-0.6, 0.7],[0.5,-0.5],[0.9, 0.2],[0.2, 1.0]])
AB   = np.array([[0.7, 0.1],[-0.2, 0.5]])
sig2B= np.array([0.30,0.50,0.40,0.60,0.35,0.45])
yB, fB = sim(20260215, 250, 6, LamB, AB, sig2B)

np.savetxt("fixtureA.csv", yA, delimiter=",",
           header=",".join(f"y{i+1}" for i in range(5)), comments="")
np.savetxt("fixtureA_truefactor.csv", fA, delimiter=",", header="f1", comments="")
np.savetxt("fixtureB.csv", yB, delimiter=",",
           header=",".join(f"y{i+1}" for i in range(6)), comments="")
np.savetxt("fixtureB_truefactor.csv", fB, delimiter=",", header="f1,f2", comments="")

# fixture A with NaNs (for missing-data test): scattered holes
yAm = yA.copy()
rng = np.random.default_rng(777)
miss_idx = [(5,0),(5,1),(40,2),(41,2),(42,2),(100,3),(150,0),(150,1),(150,2),(150,3),(150,4),(199,4)]
for (t,j) in miss_idx: yAm[t,j] = np.nan
np.savetxt("fixtureA_miss.csv", yAm, delimiter=",",
           header=",".join(f"y{i+1}" for i in range(5)), comments="")
json.dump({"miss_idx": miss_idx}, open("fixtureA_miss_idx.json","w"))

meta = {
 "A": {"T":200,"N":5,"r":1,"seed":20260214,"Lam":LamA.tolist(),"A":AA.tolist(),"sig2":sig2A.tolist(),
       "col_means":yA.mean(0).tolist(),"col_sds":yA.std(0,ddof=1).tolist(),
       "checksum_sum":float(yA.sum()),"checksum_sumsq":float((yA**2).sum())},
 "B": {"T":250,"N":6,"r":2,"seed":20260215,"Lam":LamB.tolist(),"A":AB.tolist(),"sig2":sig2B.tolist(),
       "col_means":yB.mean(0).tolist(),"col_sds":yB.std(0,ddof=1).tolist(),
       "checksum_sum":float(yB.sum()),"checksum_sumsq":float((yB**2).sum())},
}
json.dump(meta, open("fixture_meta.json","w"), indent=1)
print(json.dumps(meta, indent=1))
