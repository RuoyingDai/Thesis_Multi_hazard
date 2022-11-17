library("readxl") # snachala nuzhno ustanivit etot paket inache budet oshibka; etot dlya zagruzki excel dannyh
#library("ggplot2") # snachala nuzhno ustanivit etot paket inache budet oshibka; etot dlya vizualizacyi 
library("ggcorrplot")
library("car")
library("MASS")

setwd("/Users/yuliya/Thesis_Multi_hazard/main")
data <- read.csv("data/v27.csv",stringsAsFactors=FALSE)


# Regressiya na dannyh stud; ee mozhno zapysat dvumya ekvivalentnymy sposobami
model_fit_stud  <- lm(formula = Y ~  X1 + X2 + X3 + X4 + X5 + X6, data=stud )  # with all the independent variables in the dataframe
model_fit_stud  <- lm(formula = Y ~  .   , data=stud )  # with all the independent variables in the dataframe
summary(model_fit_stud)

# Regressiya na dannyh cars
model_fit_car <- lm(formula = price ~  .   , data=cars)  # with all the independent variables in the dataframe
summary(model_fit_car) 

# Naidem correlyaciyu
r=cor(stud)
ggcorrplot(r,lab=T,lab_size=3)

# Naidem correlyaciyu
r=cor(cars)
ggcorrplot(r,lab=TRUE,lab_size=3)

ggcorrplot(r,lab=TRUE,lab_size=3)


# Rasschitaem vif
vif(model_fit_stud) # vse vif >8(10) --> multicollinear.

# Rasschitaem vif
vif(model_fit_car) # 3 vif >8(10) --> multicollinear.

# Stroim regressiyu poshagovym metodom dlya dannyh stud
fitAIC_stud=stepAIC(model_fit_stud)

summary(fitAIC_stud)

fitAIC_stud$coefficients

AIC(fitAIC_stud)

fit2 = lm(formula = Y ~  X2 + X6, data=stud )
summary(fit2)
summary(model_fit_stud)


model_fit_car <- lm(price ~ city + highway + engine + power + fuel + weight  , data=cars)
