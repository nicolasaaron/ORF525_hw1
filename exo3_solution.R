## ------------------------------------------------------------------------
library(tictoc)

filename <- 'macro.csv'
macro = read.csv(filename,header=T)


month = macro[,1] #Months of Data
Month = strptime(month, "%m/%d/%Y") #convert to POSIXlt (a date class)

Unrate = macro[,25] #Unemploy rates
IndPro = macro[,7] #Industrial Production Index
HouSta = macro[,49] #House start
PCE = macro[,4] #Real Personal Consumption
M2Real = macro[,67] #Real M2 Money Stock
FedFund= macro[,79] #Fed Funds Rate
CPI = macro[,107] #Consumer Price Index
SPY = macro[,75] # S&P 500 index

DIndPro = diff(log(IndPro)) # changes of IndPro
DPCE = diff(log(PCE)) # changes of PCE
DM2 = diff(log(M2Real)) # chances of M2 stock
DCPI = diff(log(CPI)) # changes of CPI
DSPY = diff(log(SPY)) # log-returns of SP500

n = nrow(macro)
Y = log(PCE[2:n])  #personal consumption expediture
lag1_PCE = log(PCE[1:n-1])
lag1_Unrate = Unrate[1:n-1]
X = cbind(lag1_PCE, lag1_Unrate, DIndPro, DM2, DCPI, DSPY, HouSta[2:n], FedFund[2:n]) #present data
colnames(X) = list("lag1_logPCE", "lag1_Unrate", "DIndPro", "DM2","DCPI", "DSPY", "HouSta", "FedFund") #give covariates names


## ------------------------------------------------------------------------
#Learning/training and testing sets (last ten years)

n = length(Y)
Y.L = Y[1:(n-120)] #learning set
Y.T = Y[(n-119):n] #testing set
X.L = X[1:(n-120),] #learning set
X.T = X[(n-119):n,] #testing set
#Putting them as data frames
data_train = data.frame(Unrate=Y.L, X.L) #give Y.L the name Unrate.
data_test = data.frame(X.T)


## ------------------------------------------------------------------------
#Least-squares fit
fitted=lm(Unrate ~ ., data=data_train) #fit model using learning 
summary(fitted)


## ------------------------------------------------------------------------
# out of sample R2
Y.pred.fitted = predict(fitted, newdata= data_test)
SS.total <- sum( (Y.T- mean(Y.L))^2)  # mean square of "testing label - mean (training labels)"
SS.residual <- sum( (Y.T - Y.pred.fitted)^2)
1 - SS.residual / SS.total


## ------------------------------------------------------------------------
#########################
# part b: variable selection
############################
M <- step(fitted, data=data_train, direction="backward")
summary(M)

## ------------------------------------------------------------------------
#################
# part c
################
Y.pred = predict(M, newdata = data_test)
RMSE = sqrt(mean( (Y.pred - Y.T)^2))
MADE = mean(abs(Y.pred - Y.T))

cat('root mean squared error:', RMSE, '\nmean absolute deviation error:', MADE)


## ------------------------------------------------------------------------
###############
# part d
###############
M.values = M$fitted.values #extract fitted values
residuals = M$residuals #extract residuals
std.res = ls.diag(M)$std.res #standardized residuals

## ------------------------------------------------------------------------
#pdf("Fig_exo3_part_d_1.pdf", width=8, height=2, pointsize=10)

par(mfrow = c(1,2), mar=c(2, 4, 1.5,1)+0.1, cex=0.8)
plot(Month[(n-119):n], Y.T, type="l", col="red", lwd=2) #actual values
lines(Month[(n-119):n], Y.pred, lty=2, col="blue") #predicted values

plot(Month[1:(n-120)], Y.L, type="l", col="red", lwd=2) #actual
lines(Month[1:(n-120)], M.values, lty=2, col="blue") #fitted
title("(b) Fitted and actual ECP")

## ------------------------------------------------------------------------
# QQ plot

#pdf("Fig_exo3_part_d_2.pdf", width=8, height=4, pointsize=10)
par(mfrow = c(2,2), mar=c(2, 4, 1.5,1)+0.1, cex=0.8)

plot(Month[1:(n-120)], residuals, type="l", col="red", lwd=2) #residuals
title("(a) Time series plot of residuals")

plot(M.values, residuals, pch="*", col="red")
title("(b) Fitted versus residuals")


plot(Month[1:(n-120)], std.res, type="l", col="blue", lwd=2) #residuals
title("(c) Standardized residuals")

qqnorm(std.res, col="blue", main="(d) Q-Q plot for std.res")
qqline(std.res, col="red")

## ------------------------------------------------------------------------
##########################################
# part e Gaussian Kernel ridge regression
##########################################

require(kernlab)
require(Matrix)

krr.learn <- function (data, kernel, y, lambda) {
  K <- kernelMatrix(kernel, data)
  N <- nrow(K)
  alpha <- chol2inv(chol(K + diag(lambda, N))) %*% y
  #alpha <- solve(Matrix(K + diag(lambda, N))) %*% y
  
  s <- svd(K)
  return(list(data = data, kernel = kernel, alpha = alpha, s= s))
}

krr.predict <- function (new_data, krr) {
  k <- kernelMatrix(krr$kernel, new_data, krr$data)
  return(k %*% krr$alpha)
}

cv.krr <- function(dataset, kernel, y, lambda, folds = 5)
{
  # equally split the data into n_folds, shuffle the fold_index
  set.seed(0)
  fold_idx = sample(rep(1:folds, length.out = nrow(dataset)))
  
  mean_pred_error <- rep(0,folds)
  R2 <- rep(0, folds)
  for (k in 1:folds)
  { 
    #cat('fold number:', k, '\n')
    # extract subsamples
    current_fold = which(fold_idx == k)
    X_train = dataset[ -current_fold,]
    Y_train = y[-current_fold]
    
    X_test = dataset[ current_fold,]
    Y_test = y[current_fold]
  
    #tic('krr with time: ') 
    krr = krr.learn(X_train, kernel, Y_train, lambda)
    #toc()
    pred = krr.predict(X_test, krr)
    
    mean_pred_error[k] <- mean( (Y_test - pred)^2 )
    
    SS.total <- sum((Y_test - mean(Y_train))^2)
    SS.residual <- sum( (Y_test - pred)^2)
    R2[k] <- 1 - SS.residual / SS.total
    
    #cat('mean predictin error on test data:', mean_pred_error[k], '\n\n')
  }
  return(list(cv=mean(mean_pred_error), mse=mean_pred_error, R2=R2))
}

#standardize values of different variables
preprocessing <- function(dataset, mu, std)
{ 
  for (i in 1:ncol(dataset)) # except the first column
  { 
    dataset[,i] = data.matrix( ( dataset[,i]- mu[i] ) / std[i] )
  }
  return(data.frame(dataset))
}



## ------------------------------------------------------------------------

#(optional: for standarization)
# standardize the data
mu_train = colMeans(X.L)
std_train = apply(X.L, 2, sd)

X.L <- data.matrix(preprocessing(X.L, mu_train, std_train))
X.T <- data.matrix(preprocessing(X.T, mu_train, std_train))

# kernel parameters
gamma = 0.01/2
my_kernel = rbfdot(sigma = gamma)

# lambda for ridge penalty
N_lambda = 20
lambda_list = 10^{seq(-4,0,length = N_lambda)}

# number of folds for cross-validation
n_folds = 5

# perform Kernel ridge regression
mean_r2 = rep(0, N_lambda)
mean_mse = rep(0, N_lambda)
for (i in 1:N_lambda)
{
  lambda = lambda_list[i]
  cat('\n lambda=',lambda, '\t,')
  
  #tic('cost time=')
  result = cv.krr(dataset =data.matrix(X.L), kernel= my_kernel, y = Y.L, lambda= lambda, folds = n_folds)
  #toc()
  
  mean_r2[i] = mean(result$R2)
  cat('\t mean R2:', mean_r2[i])
  mean_mse[i] = mean(result$mse)
}

cat('\nlambda:', lambda_list)
cat('\n\nmean r2 for different lambda:\t', mean_r2)
cat('\n\nmean mse for different lambda:\t', mean_mse)


## ------------------------------------------------------------------------
#pdf("Fig_exo3_part_e.pdf", width=8, height=4, pointsize=10)

par(mar = c(4, 5, 2, 5) + 0.5)  # Leave space for z axis
plot(lambda_list, mean_r2, pch=16, type='b', yaxt='n', xlab="lambda", ylab="",log='x', col='blue', main='5 folds cv')
mtext("lambda", side=2, col="blue", line=4)
axis(side=2, at = pretty(range(mean_r2)), col='blue', col.axis='blue', las=1)
box()

par(new = TRUE)
plot(lambda_list, mean_mse, pch = 15, type = "b", axes = FALSE, bty = "n", xlab = "", ylab = "", log='x', col='red')
mtext("MSE",side=4,col="red",line=4) 
axis(side=4, at = pretty(range(mean_mse)), col='red',col.axis = 'red', las=1)



legend("topleft",legend=c("mean R2","mean MSE"),
  text.col=c("blue","red"),pch=c(16,15),col=c("blue","red"))



## ------------------------------------------------------------------------
# pick the lambda with the minimum mean mse
idx = which.min(mean_mse)
best_lambda = lambda_list[idx]
cat('best lambda', best_lambda, '\n')

gamma = 0.01/2
my_kernel = rbfdot(sigma = gamma)

# KRR
tic('\nkrr with time: ') 
krr = krr.learn(X.L, my_kernel, Y.L, best_lambda)
toc()


# in sample RMSE and R2
Y.pred.krr.in <- krr.predict(X.L, krr)
rmse.krr.in <- sqrt( mean((Y.pred.krr.in - Y.L)^2))
SS.total <- sum((Y.L - mean(Y.L))^2 )
SS.residual <- sum( (Y.L - Y.pred.krr.in)^2 )
R2.krr.in <- 1- SS.residual / SS.total

cat('\nin sample RMSE:\t', rmse.krr.in)
cat('\nin sample R2:\t', R2.krr.in)


# prediction
Y.pred.krr.out = krr.predict(X.T, krr)
# compute mean squared errors
rmse.krr = sqrt( mean((Y.pred.krr.out - Y.T)^2))

# compute out of sample R2
SS.total =  sum((Y.T - mean(Y.L))^2)
SS.residual = sum( (Y.T - Y.pred.krr.out)^2 )
R2.krr=  1- SS.residual / SS.total

cat('\nout of sample RMSE:\t', rmse.krr)
cat('\nout of sample R2:\t', R2.krr)


## ----eval=FALSE----------------------------------------------------------
## library(knitr)
## purl(input='exo3_solution.Rmd', output='exo3_solution.R')
## 

