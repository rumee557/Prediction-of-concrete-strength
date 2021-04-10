#MATH 8050 Project
#Rumi Shrestha, Arup Bhattacharya

rm(list=ls())
setwd("~/summer sem/Data analysis/Project")

library(leaps) #all subsets
library(faraway) #this will be used to check adjusted r square
library(MASS) #this has the step AIC function
library(boot) #contains tools to do cross validation
library(caret) 
library(glmnet) 
library(data.table)
library(ggplot2)
library(ggcorrplot)
library(reshape2)#for correlation heat matrix

options(max.print=1000000)
set.seed(7)
concrete_data<- read.table("Concrete_Data.txt",sep="\t",header=TRUE)
names(concrete_data) <- c("Cement","Slag","Fly Ash","Water",
                          "Superplast","CA","FA",
                          "Age","Comp_Str")

#DATA SPLITTING####
train.rows<- createDataPartition(y= concrete_data$Comp_Str, 
                                 p=0.5, list = FALSE)
train.data<- concrete_data[train.rows,] # 50% data goes in here

names(train.data) <- c("Cement","Slag","Fly Ash","Water",
                       "Superplast","CA","FA",
                       "Age","Comp_Str")
predict.data <- concrete_data[-train.rows,]
names(predict.data) <- c("Cement","Slag","Fly Ash","Water",
                         "Superplast","CA","FA",
                         "Age","Comp_Str")

#EXPLORATORY DATA ANALYSIS####

#Boxplot
x11()
boxplot(train.data[,-c(9)])#removing the columns not interested in

#Scatterplots
x11()
pairs(train.data)

#Correlation
cor(train.data)
cormat <- round(cor(train.data),4)
head(cormat)
x11()
ggcorrplot(cormat,hc.order=TRUE,type="lower",lab=TRUE)

#TRANSFORMATION OF VARIABLES####
###Fit a first-order model
model_1 <- lm(train.data$Comp_Str~.,data=train.data)
x11()
plot(x=model_1$fit, y=model_1$res,xlab="Fit",ylab="Residuals"
     ,main="Residual vs fitted values plot before transformation",pch=20)
abline(h=0)

x11()
boxcox(model_1)

#Use square-root-transform response
sqrt_Comp_Str <- sqrt(train.data$Comp_Str)
y<-sqrt_Comp_Str
model_2 <- lm(sqrt_Comp_Str ~.-Comp_Str,data=train.data)
x11()
plot(x=model_2$fit, y=model_2$res,xlab="Fit",ylab="Residuals",
     main="Residuals vs fitted values plot after transformation",pch=20)
abline(h=0)


summary(model_2)
train.data$Age <- log(train.data$Age)


#Centering and scaling the variables
train.data$sqrt_CS <- sqrt_Comp_Str
train.data.original <- train.data
for(i in 1:length(train.data)){
  train.data[,i] <- (train.data[,i]-mean(train.data[,i]))/(sd(train.data[,i])*sqrt(nrow(train.data)-1))
}

model_2 <- lm(sqrt_CS ~.-Comp_Str,data=train.data)
x11()
plot(x=model_2$fit, y=model_2$res,xlab="Fit",ylab="Res",
     main="Residuals vs fitted values plot after transformation",pch=20)
abline(h=0)

#REGRESSION ANALYSIS AND REDUCTION OF PREDICTOR VARIABLES####
###OLS REGRESSION
#Checkout adjusted R2
x.mat <- model.matrix(model_2)[,-1]#ignore the intercept column
checks <- leaps(x.mat,train.data$sqrt_CS,method="adjr2")
maxadjr(checks,best=10) #This function is in the faraway package
cor(train.data)

#Forward, backward, and stepwise variable selection
null <- lm(train.data$sqrt_CS~1,data=train.data)
full <- model_2

stepAIC(null,scope=list(lower=null,upper=full),direction="forward",trace=FALSE)

stepAIC(full,scope=list(lower=full,upper=null),direction="backward",trace=FALSE)

stepAIC(full,scope=list(lower=full,upper=null),direction="both",trace=FALSE)

#Let's check BIC
all.subsets <- regsubsets(sqrt_CS~.-Comp_Str,data=train.data)
#This does an exhaustive search

summary(all.subsets)
summary(all.subsets)$bic

#BIC favored simpler model
mod_LR <- glm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`,data=train.data)
summary(mod_LR)

#Check for multicollinearity
dat.mat <- as.matrix(train.data.original[,-9:-10])
vif(dat.mat)

###RIDGE REGRESSION
#cv.glmnet expects a matrix of predictors, not a data frame
set.seed(7)
x_ridge_fit<- as.matrix(train.data[,-c(9,10)])
y_ridge_fit<- as.matrix(train.data[,c(10)])
mod_ridge_fit <- cv.glmnet(x_ridge_fit,y_ridge_fit,
                       type.measure = "mse",alpha=0,family="gaussian")
mod_ridge_fit$lambda.1se
x11()
plot(mod_ridge_fit)
best_lambda_ridge=mod_ridge_fit$lambda.1se
ridge_coef<- 
  mod_ridge_fit$glmnet.fit$beta[,mod_ridge_fit$glmnet.fit$lambda==best_lambda_ridge]

#ELASTIC NET REGRESSION
set.seed(7)
x_elasticNet_fit<- as.matrix(train.data[,-c(9,10)])
y_elasticNet_fit<- as.matrix(train.data[,c(10)])
mod_elasticNet_fit <- cv.glmnet(x_elasticNet_fit,y_elasticNet_fit,
                           type.measure = "mse",alpha=0.5,family="gaussian")
mod_elasticNet_fit$lambda.1se
x11()
plot(mod_elasticNet_fit)
best_lambda_ElasticNet=mod_elasticNet_fit$lambda.1se
elasticNet_coef<- 
  mod_elasticNet_fit$glmnet.fit$beta[,mod_elasticNet_fit$glmnet.fit$lambda==best_lambda_ElasticNet]

#LASSO REGRESSION
set.seed(7)
x_lasso_fit<- as.matrix(train.data[,-c(9,10)])
y_lasso_fit<- as.matrix(train.data[,c(10)])
mod_lasso_fit <- cv.glmnet(x_lasso_fit,y_lasso_fit,
                                type.measure = "mse",alpha=1,family="gaussian")
mod_lasso_fit$lambda.1se
x11()
plot(mod_lasso_fit)
best_lambda_lasso=mod_lasso_fit$lambda.1se
lasso_coef<- 
  mod_lasso_fit$glmnet.fit$beta[,mod_lasso_fit$glmnet.fit$lambda==best_lambda_lasso]

#MODELS OBTAINED FROM ALL FOUR REGRESSION ANALYSIS####
mod_LR <- glm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`,data=train.data)
mod_ridge <- glm(sqrt_CS~.-Comp_Str,data=train.data)
mod_elasticNet <- glm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`+Superplast+FA
                      ,data=train.data)
mod_lasso <- glm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`+Superplast
                 ,data=train.data)



#COMPARE MODELS####

#OLS LR
x11()
plot(x=mod_LR$fit, y=mod_LR$res,xlab="Fit",ylab="Res",
     main="Residuals vs fitted values plot for OLS Linear Model",pch=20)
abline(h=0)

x11()
qqnorm(mod_LR$res,main="Normal Q-Q Plot for OLS Linear Model",pch=20)
qqline(mod_LR$res)

#leverage points
x.mat <- model.matrix(mod_LR)
lev<- hat(x.mat)
x11()
plot(lev,ylab="Leverages"
     ,main="Index plot of Leverages for OLS Linear Model",pch=20)
abline(h=2*7/nrow(train.data))
#identify(1:nrow(train.data),lev,1:nrow(train.data))

#influential points
cook<- cooks.distance(mod_LR)
x11()
plot(cook,xlab="Case",ylab="Cook's distance for OLS Linear Model",pch=20)
segments(1:nrow(train.data),0,1:nrow(train.data),cook)
#identify(1:nrow(train.data),cook,1:nrow(train.data))
qf(0.1,7,nrow(train.data)-7)

#RIDGE
x11()
plot(x=mod_ridge$fit, y=mod_ridge$res,xlab="Fit",ylab="Res",
     main="Residuals vs fitted values plot for Ridge Model",pch=20)
abline(h=0)

x11()
qqnorm(mod_ridge$res,main="Normal Q-Q Plot for Ridge Model",pch=20)
qqline(mod_ridge$res)

#leverage points
x.mat <- model.matrix(mod_ridge)
lev<- hat(x.mat)
x11()
plot(lev,ylab="Leverages",main="Index plot of Leverages for Ridge Model",pch=20)
abline(h=2*7/nrow(train.data))
#identify(1:nrow(train.data),lev,1:nrow(train.data))

#influential points
cook<- cooks.distance(mod_ridge)
x11()
plot(cook,xlab="Case",ylab="Cook's distance for Ridge Model",pch=20)
segments(1:nrow(train.data),0,1:nrow(train.data),cook)
#identify(1:nrow(train.data),cook,1:nrow(train.data))
qf(0.1,7,nrow(train.data)-7)

#ELASTIC NET
x11()
plot(x=mod_elasticNet$fit, y=mod_elasticNet$res,xlab="Fit",ylab="Res",
     main="Residuals vs fitted values plot for ElasticNet Model",pch=20)
abline(h=0)

x11()
qqnorm(mod_elasticNet$res,main="Normal Q-Q Plot for ElasticNet Model",pch=20)
qqline(mod_elasticNet$res)

#leverage points
x.mat <- model.matrix(mod_elasticNet)
lev<- hat(x.mat)
x11()
plot(lev,ylab="Leverages",
     main="Index plot of Leverages for ElasticNet Model",pch=20)
abline(h=2*7/nrow(train.data))
#identify(1:nrow(train.data),lev,1:nrow(train.data))

#influential points
cook<- cooks.distance(mod_elasticNet)
x11()
plot(cook,xlab="Case",ylab="Cook's distance for ElasticNet Model",pch=20)
segments(1:nrow(train.data),0,1:nrow(train.data),cook)
#identify(1:nrow(train.data),cook,1:nrow(train.data))
qf(0.1,7,nrow(train.data)-7)

#LASSO
x11()
plot(x=mod_lasso$fit, y=mod_lasso$res,xlab="Fit",ylab="Res",
     main="Residuals vs fitted values plot for Lasso Model",pch=20)
abline(h=0)

x11()
qqnorm(mod_lasso$res,main="Normal Q-Q Plot for Lasso Model",pch=20)
qqline(mod_lasso$res)

#leverage points
x.mat <- model.matrix(mod_lasso)
lev<- hat(x.mat)
x11()
plot(lev,ylab="Leverages",main="Index plot of Leverages for Lasso Model",pch=20)
abline(h=2*7/nrow(train.data))
#identify(1:nrow(train.data),lev,1:nrow(train.data))

#influential points
cook<- cooks.distance(mod_lasso)
x11()
plot(cook,xlab="Case",ylab="Cook's distance for Lasso Model",pch=20)
segments(1:nrow(train.data),0,1:nrow(train.data),cook)
#identify(1:nrow(train.data),cook,1:nrow(train.data))
qf(0.1,7,nrow(train.data)-7)

#quite similar normal Q-Q plots

#Check models based on cross-validation score
#find loo-cv
loo.cv.LR <- cv.glm(train.data,mod_LR)
loo.cv.LR$delta[1] #Get the cross validation score

loo.cv.ridge <- cv.glm(train.data,mod_ridge)
loo.cv.ridge$delta[1] #Get the cross validation score

loo.cv.elasticNet <- cv.glm(train.data,mod_elasticNet)
loo.cv.elasticNet$delta[1] #Get the cross validation score

loo.cv.lasso <- cv.glm(train.data,mod_lasso)
loo.cv.lasso$delta[1] #Get the cross validation score

#COMPARE OLD MODEL WITH NEW MODEL####
#For the model using validating data
sqrt_Comp_Str <- sqrt(predict.data$Comp_Str)

predict.data$sqrt_CS <- sqrt_Comp_Str

predict.data$Age <- log(predict.data$Age)
predict.data.original <- predict.data
for(i in 1:length(predict.data)){
  predict.data[,i] <- (predict.data[,i]-mean(predict.data[,i]))/
    (sd(predict.data[,i])*sqrt(nrow(predict.data)-1))
}

#OLS LR
mod_LR_predict <- glm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`,data=predict.data)
summary(mod_LR_predict)
predict.LR <- predict(mod_LR, newdata= predict.data[, c(1,2,3,4,8)])

LR_rmse <- sqrt(mean((predict.data$sqrt_CS-predict.LR)^2))
mod_LR_predict_unstandard <- glm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`
                                 ,data=predict.data.original)


#RIDGE
mod_ridge_predict <- glm(sqrt_CS~.-Comp_Str,data=predict.data)
x_ridge_predict<- as.matrix(predict.data[,-c(9,10)])
y_ridge_predict<- as.matrix(predict.data[,c(10)])
mod_ridge_predict <- predict(cv.glmnet(x_ridge_fit,y_ridge_fit)
                             ,s=mod_ridge_fit$lambda.1se,newx=x_ridge_predict)
ridge_rmse <- sqrt(mean((y_ridge_predict-mod_ridge_predict)^2))

#LASSO
mod_lasso_predict <- glm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`+Superplast
                         ,data=predict.data)
x_lasso_predict<- as.matrix(predict.data[,-c(9,10)])
y_lasso_predict<- as.matrix(predict.data[,c(10)])
mod_lasso_predict <- predict(mod_lasso_fit,s=mod_lasso_fit$lambda.1se
                             ,newx=x_lasso_predict)
lasso_rmse <- sqrt(mean((y_lasso_predict-mod_lasso_predict)^2))

#ELASTIC NET
mod_elasticNet_predict <- glm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`+Superplast
                              +FA,data=predict.data)
x_elasticNet_predict<- as.matrix(predict.data[,-c(9,10)])
y_elasticNet_predict<- as.matrix(predict.data[,c(10)])
mod_elasticNet_predict <- predict(mod_elasticNet_fit
                                  ,s=mod_elasticNet_fit$lambda.1se
                                  ,newx=x_elasticNet_predict)
elasticNet_rmse <- sqrt(mean((y_elasticNet_predict-mod_elasticNet_predict)^2))

#CHECK INTERACTION#### 
x11()
plot(mod_LR$res~I(train.data$Cement*train.data$Slag)
     ,main="Residuals vs Cement and Slag Interation",pch=20)
abline(h=0)

x11()
plot(mod_LR$res~I(train.data$Cement*train.data$`Fly Ash`)
     ,main="Residuals vs Cement and Fly Ash Interation",pch=20)
abline(h=0)

x11()
plot(mod_LR$res~I(train.data$Cement*train.data$Water)
     ,main="Residuals vs Cement and Water Interation",pch=20)
abline(h=0)

x11()
plot(mod_LR$res~I(train.data$Cement*train.data$Age)
     ,main="Residuals vs Cement and Age Interation",pch=20)
abline(h=0)

x11()
plot(mod_LR$res~I(train.data$Slag*train.data$'Fly Ash')
     ,main="Residuals vs Slag and Fly Ash Interation",pch=20)
abline(h=0)

x11()
plot(mod_LR$res~I(train.data$Slag*train.data$Water)
     ,main="Residuals vs Slag and Water Interation",pch=20)
abline(h=0)

x11()
plot(mod_LR$res~I(train.data$Slag*train.data$Age)
     ,main="Residuals vs Slag and Age Interation",pch=20)
abline(h=0)

x11()
plot(mod_LR$res~I(train.data$`Fly Ash`*train.data$Water)
     ,main="Residuals vs Fly Ash and Water Interation",pch=20)
abline(h=0)

x11()
plot(mod_LR$res~I(train.data$`Fly Ash`*train.data$Age)
     ,main="Residuals vs Fly Ash and Age Interation",pch=20)
abline(h=0)

x11()
plot(mod_LR$res~I(train.data$Water*train.data$Age)
     ,main="Residuals vs Water and Age Interation",pch=20)
abline(h=0)

cor(train.data[,c(1,2,3,4,8,10)])

#Get the model matrix for the model (design matrix, X)
#this lengthy procedure as a check

mod_LR_original <- glm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`
                       ,data=train.data.original)
(xmat_LR <- model.matrix(mod_LR_original))
(y_LR <- matrix(train.data.original$sqrt_CS))

xtx <- t(xmat_LR)%*%xmat_LR
xty <- t(xmat_LR)%*%y_LR

beta_hat <- solve(xtx,xty)

J <- matrix(1,nrow(y_LR),nrow(y_LR)) #J matrix
n <- nrow(y_LR) 
yty <- t(y_LR)%*% y_LR
nytjy <- (t(y_LR)%*%J%*%y_LR)/n
SST <- yty-nytjy #total sum of squares
SSE <- yty-t(beta_hat)%*%t(xmat_LR)%*%y #error sum of squares
SSR <- SST-SSE #regression sum of squares
df_sse <- n-nrow(beta_hat) 
df_ssr <- nrow(beta_hat)-1
MSR <- SSR/df_ssr
MSE <- SSE/df_sse
F_statistic <- MSR/MSE
p_value <- pf(F_statistic,df_ssr,df_sse,lower.tail = FALSE)
R2 <- 1-SSE/SST
Ra2 <- 1-((n-1)/df_sse)*SSE/SST



#SUMMARY OF RESULTS####
summary(mod_LR)#summary of the accepted standardized model
mod_LR_non_std <- glm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`,
                      data=train.data.original)
#summary of the accepted non-standardized model
#anova
anova_LR <- aov(mod_LR_non_std)
summary(anova_LR)
#Coefficient of determination
summary(lm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`,data=train.data))$r.squared
#Adjusted coefficient of determination
summary(lm(sqrt_CS~Age+Cement+Water+Slag+`Fly Ash`
           ,data=train.data))$adj.r.squared





