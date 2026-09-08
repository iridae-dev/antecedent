# Executing external oracle: R survival 3.8.6, Breslow ties, left limits.
# Run from repository root. Output columns are consumed by freeze.py.
stopifnot(as.character(packageVersion("survival")) == "3.8.6")
library(survival)
options(digits=17)
i <- 1:180
z <- sin(i*.73)
a <- cos(i*.37)
y <- -log(((i*37) %% 181 + .5)/182)/exp(.25*a)
c <- -log(((i*61) %% 179 + .5)/180)/exp(.65*z-.35*a)
time <- round(pmin(y,c),2)
event <- as.numeric(y<=c)
fit <- coxph(Surv(time,1-event)~a+z, ties="breslow", control=coxph.control(eps=1e-12,iter.max=100))
bh <- basehaz(fit,centered=FALSE)
h <- vapply(time,function(t) sum(diff(c(0,bh$hazard))[bh$time<t]),numeric(1))
g <- exp(-h*exp(coef(fit)[1]*a+coef(fit)[2]*z))
write.table(data.frame(time,event,a,z,survival=g,weight=event/g),
 "conformance/response/conditional_ipcw/oracle.csv",row.names=FALSE,sep=",")
write.table(coef(fit),"conformance/response/conditional_ipcw/coefficients.txt",row.names=FALSE,col.names=FALSE)
