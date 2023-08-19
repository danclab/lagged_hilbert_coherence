library('lme4')
library('car')
library('emmeans')
library('ggplot2')
df<-read.csv('../output/sim_burst_duration.csv')
df$trial<-as.factor(df$trial)
df$freq<-as.factor(df$freq)

m1<-lm(x0 ~ freq*burst_d, data=df)
Anova(m1, type = 3)

m2<-lm(x0 ~ freq*burst_d_c, data=df)
Anova(m2, type = 3)

aic1 <- AIC(m1)
aic2 <- AIC(m2)
print(aic2-aic1)

m1<-lm(k ~ freq*burst_d, data=df)
Anova(m1, type = 3)

m2<-lm(k ~ freq*burst_d_c, data=df)
Anova(m2, type = 3)

aic1 <- AIC(m1)
aic2 <- AIC(m2)
print(aic1-aic2)


df<-read.csv('../output/sim_burst_number.csv')
df$trial<-as.factor(df$trial)
df$freq<-as.factor(df$freq)

m<-lm(x0 ~ freq*burst_n, data=df)
Anova(m, type = 3)

m<-lm(k ~ freq*burst_n, data=df)
Anova(m, type = 3)
