import ast
CORE_DF = {"__init__","_initialize_loadings","_initialize_exog","_initialize_error_cov",
 "_initialize_error_cov_diagonal","_initialize_error_cov_unstructured","_initialize_factor_transition",
 "_initialize_error_transition","_initialize_error_transition_white_noise","_initialize_error_transition_var",
 "_initialize_error_transition_individual","start_params","param_names","state_names","transform_params",
 "untransform_params","update","_validate_can_fix_params","clone","_res_classes"}
CORE_MQ = {"__init__","construct_endog","clone","_default_initialization","_get_endog_names","state_names",
 "param_names","start_params","transform_params","untransform_params","update","loglike_constant",
 "loading_constraints","fit","fit_em","_em_iteration","_em_expectation_step","_em_maximization_step",
 "_em_maximization_obs_nonmissing","_em_maximization_obs_missing","smooth","filter",
 "factors_ix","factors","factors_ar","factors_L1","factors_L1_5","_apply_factor_multiplicities",
 "_construct_endog_factor_map","factors_L1_5_ix","idio_ar_L1","idio_ar_M","idio_ar_Q","idio_ar_Q_ix",
 "endog_factor_iloc","__getitem__"}
for path, core, label in [
  ("/usr/local/lib/python3.13/dist-packages/statsmodels/tsa/statespace/dynamic_factor.py", CORE_DF, "dynamic_factor.py"),
  ("/usr/local/lib/python3.13/dist-packages/statsmodels/tsa/statespace/dynamic_factor_mq.py", CORE_MQ, "dynamic_factor_mq.py")]:
    src=open(path).read(); tree=ast.parse(src)
    tot=len(src.splitlines()); c=b=0; crows=[]; brows=[]
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            for s in n.body:
                if isinstance(s, ast.FunctionDef):
                    ds=ast.get_docstring(s); dl=len(ds.splitlines()) if ds else 0
                    span=s.end_lineno-s.lineno+1; code=span-dl
                    if s.name in core and "Results" not in n.name:
                        c+=code; crows.append((n.name,s.name,span,code))
                    else:
                        b+=code; brows.append((n.name,s.name,span,code))
    print("="*95); print(f"{label}: total file lines={tot}")
    print(f"  MODELLING CORE  (code lines, docstrings excluded) = {c}")
    for r in sorted(crows,key=lambda x:-x[3])[:12]: print(f"      {r[0]}.{r[1]:<38} span={r[2]:<5} code={r[3]}")
    print(f"  BOILERPLATE/RESULTS/PLOTTING (code lines)         = {b}")
    for r in sorted(brows,key=lambda x:-x[3])[:10]: print(f"      {r[0]}.{r[1]:<38} span={r[2]:<5} code={r[3]}")
    print(f"  core share of methods = {100*c/(c+b):.1f}%   docstring+module lines = {tot-(c+b)}")
