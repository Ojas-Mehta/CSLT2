# library(car)
library(caret)
library (ridge)
library(elasticnet)
library(glmnet)
library(leaps)
library(coefplot)
library(ggplot2)
library(np)
library(matrixStats)
library(scales)
library(corrplot)
require(kernlab) # dependency of CVST and DDR
require(CVST) # library for KRR, loads kernlab
require(KRLS)  # library for fast cross validatio

temp <- read.csv("datadir/kc_house_train_data.csv", header = TRUE)
df.train<-subset(temp,select=c(-id,-date))
df.train[] <- lapply(df.train, function(x) {if(is.factor(x)) as.numeric(as.character(x)) else x})
df.train<-as.data.frame(scale(df.train))
df.train.X<-subset(df.train,select=c(-price))
df.train.y<-df.train$price
temp<- read.csv("datadir/kc_house_test_data.csv", header = TRUE)
df.test<-subset(temp,select=c(-id,-date))
df.test[] <- lapply(df.test, function(x) {if(is.factor(x)) as.numeric(as.character(x)) else x})
df.test<-as.data.frame(scale(df.test))
df.test.X<-subset(df.test,select=c(-price))
df.test.y<-df.test$price

corrplot(cor(df.train))
corrplot(cor(df.train.X),method = 'ellipse')
ctrl <- trainControl(method = "cv",number = 10)
set.seed(21)
kk<-krls(X=df.train.X[1:5000,],y=df.train.y[1:5000], whichkernel = "gaussian",lambda = 0.00001)
kernelRegModel1<-kk
fin<-kk$avgderivatives
print('yay')
for(lambda_ in 10^seq(-4, 0, 1)){
  # hyperfolds=10
  kk<-krls(X=df.train.X[1:5000,],y=df.train.y[1:5000], whichkernel = "gaussian",lambda = lambda_)
  fin<-rbind(fin,kk$avgderivatives)
  print('yayy')
}
kernel_predict<-predict(kernelRegModel1, newdata = df.test.X)
kernel_train_predict<-predict(kernelRegModel1, newdata = df.train.X)
kernel_test_rss = sum((df.test.y - kernel_predict$fit) ^ 2)
kernel_train_rss = sum((df.train.y - kernel_train_predict$fit) ^ 2)
kernel_test_rmse = sqrt(kernel_test_rss/length(kernel_predict))
kernel_train_rmse = sqrt(kernel_train_rss/length(kernel_train_predict))
ridgeModel <- train(y=train.y, x = df.train.X,trControl = ctrl,method = 'ridge',tuneGrid = expand.grid(.lambda=0.001), metrics='RMSE',preProc = c("center", "scale"))
ridge_coeff<-predict(ridgeModel$finalModel, type = "coefficients")
ridge_predict<-predict(ridgeModel, newdata = df.test.X)
ridge_train_predict<-predict(ridgeModel, newdata = df.train.X)
ridge_test_rss = sum((df.test.y - ridge_predict) ^ 2)
ridge_train_rss = sum((df.train.y - ridge_train_predict) ^ 2)
ridge_test_rmse = sqrt(ridge_test_rss/length(ridge_predict))
ridge_train_rmse = sqrt(ridge_train_rss/length(ridge_train_predict))
BackwardStepwiseModel <- train(y=df.train.y, x = df.train.X,trControl = ctrl,method = 'leapBackward',tuneGrid = data.frame(nvmax = 18), metrics='RMSE',preProc = c("center", "scale"))
BackwardStepwiseModel.summary <- summary(BackwardStepwiseModel)
selected.subset<-df.train.X[BackwardStepwiseModel.summary$which[10,]]
selected.bestModel<-train(y=train.y, x = selected.subset,trControl = ctrl,method = 'glmnet',preProc = c("center", "scale"))
bsm_coeff<-predict(selected.bestModel$finalModel, type = "coefficients")
bsm_predict<-predict(selected.bestModel, newdata = df.test.X)
bsm_train_predict<-predict(selected.bestModel, newdata = df.train.X)
bsm_test_rss = sum((df.test.y - bsm_predict) ^ 2)
bsm_train_rss = sum((df.train.y - bsm_train_predict) ^ 2)
bsm_test_rmse = sqrt(bsm_test_rss/length(bsm_predict))
bsm_train_rmse = sqrt(bsm_train_rss/length(bsm_train_predict))


# serdtfyguhjiko
# RidgeModel <- train(y=df.train.y, x = df.train.X,trControl = ctrl,method = 'ridge',tuneGrid = expand.grid(.lambda=0.001), metrics='RMSE')
# BackwardStepwiseModel <- train(y=df.train.y, x = df.train.X,trControl = ctrl,method = 'leapBackward',tuneGrid = data.frame(nvmax = 18), metrics='RMSE',preProc = c("center", "scale"))
# trainXmat<-matrix(as.numeric(unlist(subset(df.train.X,select=c(-date,-id)))),nrow=nrow(df.train.X))
# zz <- predict(backwardStepwiseModel, newdata = df.train.y)
# train.subset<-subset(df.train.X,select=c(-id,-date))
# selected.subset<-train.subset[BackwardStepwiseModel.summary$which[10,]]
# selected.bestModel<-train(y=train.y, x = selected.subset,trControl = ctrl,method = 'glmnet',preProc = c("center", "scale"))
# plot(selected.bestModel$finalModel)
# train.data = matrix(as.numeric(unlist(df.train)),nrow=nrow(df.train))
# train.X = matrix(as.numeric(unlist(df.train.X)),nrow=nrow(df.train))
# linRidgeMod <- linearRidge(train.y ~ ., data = subset(df.train.X,select=c(-id,-date)), lambda=0.01)  # the ridge regression model
# ctrl <- trainControl(method = "repeatedcv", repeats = 2,number = 10)
# # ridgeModel1 <- train(y=train.y, x = subset(df.train.X,select=c(-id,-date)),trControl = ctrl,method = 'ridge',tuneGrid = expand.grid(.lambda=1), metrics='RMSE')
# backwardStepwiseModel <- train(y=train.y, x = subset(df.train.X,select=c(-id,-date)),trControl = ctrl,method = 'leapBackward',tuneGrid = data.frame(nvmax = 18), metrics='RMSE',preProc = c("center", "scale"))
# trainXmat<-matrix(as.numeric(unlist(subset(df.train.X,select=c(-date,-id)))),nrow=nrow(df.train.X))
# # zz <- predict(backwardStepwiseModel, newdata = df.train.y)
# train.subset<-subset(df.train.X,select=c(-id,-date))
# selected.subset<-train.subset[bsm.summary$which[10,2:19]]
# selected.bestModel<-train(y=train.y, x = selected.subset,trControl = ctrl,method = 'glmnet',preProc = c("center", "scale"))
# plot(selected.bestModel$finalModel)
# linearKernelModel <- train(y=train.y[1:100], x = subset(df.train.X,select=c(-id,-date))[1:100,],method = 'krlsPoly',trControl = ctrl, tuneGrid = expand.grid(lambda=0.0001,degree=1,derivative=TRUE))
# polyKernelModel <- train(y=train.y, x = subset(df.train.X,select=c(-id,-date)),trControl = ctrl,method = 'svmPoly',direction='backward', metrics='RMSE')
# # print(train.y)
# # train.num <- matrix(as.numeric(unlist(df.train)),nrow=nrow(df.train))
# # data_correlation <-cor(train.num)
# set.seed(21)
# ridgeModel1 <- train(y=train.y, x = subset(df.train,select=c(-date,-price)),trControl = ctrl,method = 'ridge',tuneGrid = expand.grid(lambda=0.1), metrics='RMSE',preProc = c("center", "scale"))
# zz1<-predict(ridgeModel1, newdata = subset(df.train,select=c(-date,-price)))
# coefff<-predict(ridgeModel1$finalModel, type = "coefficients")

# bw<-npregbw(ydat = train.y, xdat = subset(df.train.X,select=c(-id,-date)))
# model <- npreg(bws = bw, gradients = TRUE)
# bestCoeff<-coefff$coefficients[19,]
# coefplot(polyKernelModel$finalModel)
# library(CVST)
# d <- constructData(x=train.X[1:10,], y=train.y[1:10]) ## Structure data in CVST format
# krr_learner <- constructFastKRRLearner()   ## Build the base learner
# params = constructParams(kernel='rbfdot', sigma=10^(seq(-8,-4,length=5)), lambda=10^(seq(-12,-8,length=5)), nblocks=1) ## Function params; documentation defines lambda as '.1/getN(d)'
# cv.krr<-CV(data=d,learner=krr_learner,params=params, fold=10,verbose=TRUE)
# params.cv.krr<-cv.krr[[1]]
# krr_trained = krr_learner$learn(d, params.cv.krr)
# dTest = constructData(train.X[1:10,],y=train.y[1:10])
# ## Now to predict, format your test data, 'dTest', the same way as you did in 'd'
# pred = krr_learner$predict(krr_trained, dTest)
# kk<-krls(X=train.X[1:500,],y=train.y[1:500], whichkernel = "gaussian",lambda = 0.00001)
# fin<-kk$avgderivatives
# for(lambda_ in 10^seq(-4, 0, 1)){
# kk<-krls(X=train.X[1:500,],y=train.y[1:500], whichkernel = "gaussian",lambda = lambda_)
# fin<-rbind(fin,kk$avgderivatives)
# }

asdr<- fin                             #for kernel reg
# asdr<- ridge_coeff$coefficients        #for ridge reg
# asdr<- t(as.matrix(bsm_coeff))[,2:12]  #for backward stepwise reg
feature_var<- colVars(as.matrix(asdr))
feature_var_sorted<- sort(feature_var, index.return=TRUE)
lambdas<-10^seq(-5, 0, 1)                  #for kernel reg
# # lambdas<-ridge_coeff$fraction                #for ridge reg
# # lambdas<-selected.bestModel$finalModel$lambda  #for backward stepwise reg
listl<-length(feature_var_sorted$ix)
tmp <- as.data.frame(as.data.frame(asdr[,c(feature_var_sorted$ix[(listl-4):listl])]))
# tmp <- as.data.frame(as.data.frame(asdr[,1:9]))
tmp$coef <- row.names(tmp)
tmp <- reshape::melt(tmp, id = "coef")
tmp$lambda <- lambdas # extract the lambda values
#
ggplot(tmp, aes(lambda, value, group = variable,color = variable, linetype = variable)) +
  geom_line() +
  scale_x_log10() +
  xlab("Lambda") +
  ylab("Coefficients") +
  # guides(color = guide_legend(title = ""),
  #        linetype = guide_legend(title = "")) +
  guides(colour=guide_legend(override.aes=list(
    colour=hue_pal()(5)))) +
  theme_bw() +
  theme(legend.key.width = unit(3,"lines"))

