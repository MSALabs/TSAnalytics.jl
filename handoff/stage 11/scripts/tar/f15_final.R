options(digits=12)
suppressMessages(library(tsDyn))
y <- log10(lynx)
m1 <- suppressWarnings(setar(y, m=2, thDelay=1))
cat("=== n bookkeeping ===\n")
cat("length(x)        =", length(m1$str$x), "\n")
cat("str$n.used       =", m1$str$n.used, "\n")
cat("getNUsed         =", tsDyn:::getNUsed(m1$str), "\n")
cat("nrow(xx)         =", nrow(m1$str$xx), "\n")
cat("length(residuals)=", length(m1$residuals), "\n")
cat("mse = SSR/n.used =", deviance(m1)/m1$str$n.used, " vs mse():", mse(m1), "\n")
cat("summary df       =", summary(m1)$df, "\n")
cat("=> tsDyn divides by the FULL series length (114), not the 112 regression rows\n")

cat("\n=== setar.boot / predict / predict_rolling ===\n")
print(args(setar.boot)); print(args(tsDyn:::predict.nlar)); print(args(predict_rolling))
set.seed(99); b <- setar.boot(m1, boot.scheme="resample")
cat("setar.boot output: class",class(b)," length",length(b)," head:",sprintf("%.8f",head(as.numeric(b),4)),"\n")
p <- predict(m1, n.ahead=5); cat("predict n.ahead=5:",paste(sprintf("%.10f",as.numeric(p)),collapse=" "),"\n")
cat("\n=== regime() accessor ===\n")
print(table(regime(m1), useNA="ifany"))
print(head(regime(m1, initVal=FALSE, timeAttr=FALSE), 5))
cat("\n=== charac_root / ar_mean ===\n")
print(charac_root(m1)); print(ar_mean(m1))

cat("\n=== write fixture JSON ===\n")
f <- function(v) paste(sprintf("%.17g", as.numeric(v)), collapse=", ")
sm <- summary(m1)
cat(sprintf('{\n "series": "log10(datasets::lynx)", "n": 114, "m": 2, "thDelay": 1, "trim": 0.15,\n "th": %s,\n "coef": [%s],\n "coefnames": ["%s"],\n "se": [%s],\n "SSR": %s,\n "mse": %s,\n "AIC": %s,\n "BIC": %s,\n "k": %d,\n "nL": 78, "nH": 34, "rows": 112,\n "RegProp": [%s]\n}\n',
  sprintf("%.17g", as.numeric(tsDyn:::getTh(coef(m1)))),
  f(head(coef(m1),-1)), paste(head(names(coef(m1)),-1),collapse='","'),
  f(sm$coefficients[,2]), sprintf("%.17g",deviance(m1)), sprintf("%.17g",mse(m1)),
  sprintf("%.17g",AIC(m1)), sprintf("%.17g",BIC(m1)), m1$k, f(m1$model.specific$RegProp)),
  file="fixture_lynx_setar.json")
cat(readLines("fixture_lynx_setar.json"), sep="\n")
write.csv(data.frame(t=1821:1934, lynx=as.numeric(lynx), log10lynx=as.numeric(y)), "fixture_lynx.csv", row.names=FALSE)
