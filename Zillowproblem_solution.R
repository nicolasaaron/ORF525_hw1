## ------------------------------------------------------------------------
library(tictoc)

filename <- 'train.data.csv'
train_data <- read.csv(filename, header=TRUE)

filename <- 'test.data.csv'
test_data <- read.csv(filename, header=TRUE)

cat(str(train_data))
train_data$zipcode <- as.factor(train_data$zipcode)
test_data$zipcode <- as.factor(test_data$zipcode)

## ------------------------------------------------------------------------

########### part a ######################
variable_list = c('price','bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot')
train_data_new <- data.frame( train_data[, variable_list])
test_data_new <- data.frame( test_data[, variable_list])

# fit lm 
fit.lm1 <- lm(price ~ ., data = train_data_new)
cat('\nin sample R2:\t', summary(fit.lm1)$r.squared, '\n')

# out-of-sample R2
fit.lm1.pred.out <- predict(fit.lm1, newdata = test_data_new)

SS.total <- sum((test_data_new$price - mean(train_data_new$price))^2)
SS.residual <- sum( (test_data_new$price - fit.lm1.pred.out)^2)
lm1.r2.out <- 1 - SS.residual / SS.total

cat('\nout of sample R2',lm1.r2.out)




## ------------------------------------------------------------------------
############part b##########################
# adding interaction terms
#########################################
variable_list = c('price','bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot')
train_data_new <- data.frame( train_data[, variable_list])
test_data_new <- data.frame( test_data[, variable_list])

# construct new training and testing data set (with interaction)
col_interact = c('bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot')
col_names <- colnames(train_data_new)

for (i in 1:3)
{
  for (j in (i+1):4)
  {
    name= col_interact[i]
    name2= col_interact[j]
    col_name = paste0(name,' * ',name2)
    #print(col_name)
      
    train_data_new[, col_name]= as.vector(as.numeric(train_data[,name])
                                         * as.numeric(train_data[,name2]))
    test_data_new[, col_name]= as.vector(as.numeric(test_data[,name])*
                                           as.numeric(test_data[,name2]))
  } 
}
print(dim(train_data_new))


# fit lm
formula = as.formula('price ~ bedrooms + bathrooms + sqft_living + sqft_lot+
                      bedrooms * bathrooms + bedrooms * sqft_living + 
                      bedrooms * sqft_lot +  bathrooms * sqft_living + 
                      bathrooms * sqft_lot + sqft_living * sqft_lot')

fit.lm2 <- lm( formula = formula, data = train_data_new)
cat('in sample R2:\t', summary(fit.lm2)$r.squared, '\n')


# out of sample prediction 
fit.lm2.pred.out <- predict(fit.lm2, newdata = test_data_new)

# out-of-sample R2
SS.total <- sum((test_data_new$price - mean(train_data_new$price))^2)
SS.residual <- sum( (test_data_new$price - fit.lm2.pred.out)^2)
lm2.r2.out <- 1 - SS.residual / SS.total
cat('out of sample R2:', lm2.r2.out)


## ------------------------------------------------------------------------
######################
# Gaussian Kernel ridge regression on full sample 
# varaibles : c('bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot') and their interactions
#
# Attention: for full training samples on 4 variables, it requires at least 5 - 20 mins !
# 
# the following are 4 helper functions:
# krr.learn, krr.predict, cv.krr, subsamples
#####################
require(kernlab)
require(Matrix)

krr.learn <- function (data, kernel, y, lambda) {
  K <- kernelMatrix(kernel, data)
  N <- nrow(K)
  #alpha <- solve(Matrix(K + diag(lambda, N))) %*% y
  alpha <- chol2inv(chol( Matrix(K + diag(lambda, N))) ) %*% y

  return(list(data = data, kernel = kernel, alpha = alpha))
}

krr.predict <- function (new_data, krr) {
  k <- kernelMatrix(krr$kernel, new_data, krr$data)
  return(k %*% krr$alpha)
}

cv.krr <- function(X, kernel, y, lambda, folds = 5)
{
  # equally split the data into n_folds, shuffle the fold_index
  set.seed(0)
  fold_idx = sample(rep(1:folds, length.out = nrow(X)))
  
  mean_pred_error <- rep(0,folds)
  R2<- rep(0,folds)
  for (k in 1:folds)
  { 
    #cat('fold number:', k, '\n')
    # extract subsamples
    current_fold = which(fold_idx == k)
    X_train = X[ -current_fold,]
    Y_train = y[ -current_fold]
    
    X_test = X[ current_fold,]
    Y_test = y[current_fold]
  
    #tic('krr with time: ') 
    krr = krr.learn(X_train, kernel, Y_train, lambda)
    #toc()
    pred = krr.predict(X_test, krr)
    
    mean_pred_error[k] <- mean( (Y_test - pred)^2 )
    #cat('mean predictin error on test data:', mean_pred_error[k], '\n\n')
    
    SS.total <- sum((Y_test - mean(Y_train))^2)
    SS.residual <- sum( (Y_test - pred)^2)
    R2[k] <- 1 - SS.residual / SS.total
  }
  return(list(cv=mean(mean_pred_error), mse=mean_pred_error, R2= R2))
}


subsamples <- function(X,num_parts,idx)
{
  # split the data and then shuffle
  X <- data.matrix(X)
  X_list = list()
  for (i in 1:num_parts)
  {
    current_idx = which(idx == i)
    X_list[[i]] = data.matrix( X[current_idx,])
  }
  return(X_list)
}


## ------------------------------------------------------------------------
######################
# preprocessing:
######################

#standardize values of different variables, except for columns : price (or the first column)
preprocessing <- function(dataset, mu, std)
{ 
  for (i in 2:ncol(dataset)) # except the first column
  { 
    dataset[,i] = data.matrix( ( dataset[,i]- mu[i] ) / std[i] )
  }
  return(data.frame(dataset))
}




## ------------------------------------------------------------------------
# standardization
mu_train = colMeans(train_data_new)
std_train = apply( data.matrix(train_data_new), 2, sd)
train_data_new <- data.frame( preprocessing( train_data_new, mu_train, std_train) )
test_data_new <- data.frame( preprocessing( test_data_new, mu_train, std_train) )

# contruct kernel
gamma = 0.01/2
my_kernel = rbfdot(sigma = gamma)

## ------------------------------------------------------------------------
##############################################
# # cross validation for a given lambda
# # parameter set
#
# gamma = 0.01/2
# lambda = 0.1
# n_folds = 5
# num_parts = 10   # number of subsampels
################################################
lambda = 0.1
#print(lambda)

# construction of subsamples (for time constraints)
# 1. eually split the data into num_parts
#   for debuging, we can try to split into 10 parts,
#   kernel ridge regression takes about 5s for each subsample
num_parts = 10
set.seed(0)
idx = sample(rep(1:num_parts, length.out = nrow(train_data_new)))
X_train =subsamples(train_data_new[,-1], num_parts, idx) 
Y_train =subsamples(train_data_new[, 'price'], num_parts, idx)

cv_subsamples = rep(0, num_parts)
cv_result_list = list()
for (i in 1:num_parts)
{
  cat('\nsubsample No', i,'\t')
  #tic('cross validation with time')
  cv_result = cv.krr(X_train[[i]], my_kernel, Y_train[[i]], 
                     lambda = lambda, folds = 5)
  #toc()
  
  cv_result_list[[i]] = cv_result
  cv_subsamples[i] = cv_result$cv
  cat('CV:',cv_result$cv)
}

cat('\n\n report cv scores (median of 10 part sub-samples):', median(cv_subsamples))

## ------------------------------------------------------------------------
###############################################
### use cross validation to find best lamda
##############################################

#N_lambda = 10
#lambda_list = 10^{seq(-4,0,length = N_lambda)}

lambda_list = c(0.001, 0.01, 0.1, 1, 10)
N_lambda = length(lambda_list)

num_parts = 10
set.seed(0)
idx = sample(rep(1:num_parts, length.out = nrow(train_data_new)))
X_train =subsamples(train_data_new[,-1], num_parts, idx) 
Y_train =subsamples(train_data_new[, 'price'], num_parts, idx)

cv_subsamples = matrix(0, nrow = N_lambda, ncol = num_parts)

for (k in 1:N_lambda)
{
  lambda = lambda_list[k]
  
  cat('\n lambda=',lambda, '\t')
  tic('cost time=')
  for (i in 1:num_parts)
  {
    #cat('\n subsample No', i,'\n')
    #tic('cross validation with time')
    cv_result = cv.krr(X_train[[i]], my_kernel, Y_train[[i]], 
                       lambda = lambda, folds = 5)
    #toc()
    
    cv_subsamples[k,i] = cv_result$cv
    #cat('CV:',cv_result$cv)
  }
  toc()
  
  cat('CV score for subsamples:\n', cv_subsamples[k,],'\n')
}

## ------------------------------------------------------------------------
########################
# output  best lambda
#######################

# pick the median as CV for a given lambda
cv_lambda = apply(cv_subsamples, 1, median)
cat('\n list of lambda values:',lambda_list)

if (num_parts > 1){
  cat('\n\n CV for each lambda (pick as median from sub-samples):', cv_lambda)
}else{
  cat('\n\n CV for 5 folds cross validation score', cv_lambda)
}

cat('\nlist of lambda values:',lambda_list)
cat('\n\nthe best lambda =', lambda_list[which.min(cv_lambda)] )

## ------------------------------------------------------------------------
#################################################
# run with the best lambda 
######################################
lambda.min = lambda_list[which.min(cv_lambda)]
#lambda = 0.1

num_parts = 10
set.seed(0)
idx = sample(rep(1:num_parts, length.out = nrow(train_data_new)))
X_train =subsamples(train_data_new[,-1], num_parts, idx) 
Y_train =subsamples(train_data_new[, 'price'], num_parts, idx)

X_test = data.matrix(test_data_new[,-1])
Y_test = data.matrix(test_data_new$price)


# 4. regression on subsamples, if num_parts = 1, then regression on full sample size
mse_krr= rep(0, num_parts)
R2_krr = rep(0,num_parts)
for (i in 1:num_parts)
{
  cat('\n\nsubsample set No:',i, '\n')
  tic('krr with time: ') 
  krr = krr.learn(X_train[[i]], my_kernel, Y_train[[i]], lambda)
  toc()
  pred = krr.predict(X_test, krr)
  
  #MSE
  mse_krr[i] = mean((pred - Y_test)^2)
  # compute out of sample R2
  SS.total =  sum((Y_test - mean(Y_train[[i]]))^2)
  SS.residual = sum( (Y_test - pred)^2 )
  R2_krr[i]= 1- SS.residual / SS.total
  
  #cat('mean prediction eror (test_data):\t', mse_krr[[i]], '\n' )
  #cat('R2 on test_data:\t', R2_krr[[i]], '\n\n' )
}


cat('\n\nout-of-sample R2 (median from subsamples)', median(R2_krr) )

## ------------------------------------------------------------------------
#################################################
# run on whole sample using the best lambda
################################################
flag_run_on_whole = TRUE
if (flag_run_on_whole)
{
lambda.min = lambda_list[which.min(cv_lambda)]
#lambda.min = 0.1

# construct traing data and testing data
X_train =data.matrix(train_data_new[,-1])
Y_train =data.matrix(train_data_new[, 'price'])
X_test = data.matrix(test_data_new[,-1])
Y_test = data.matrix(test_data_new$price)
print(dim(X_train))
# regression 

tic('krr with time: ') 
krr = krr.learn(X_train, my_kernel, Y_train, lambda= lambda.min)
toc()

#  out of sample prediction 
pred.out = krr.predict(X_test, krr)
#out of sample MSE
rmse_krr.out = sqrt( mean((pred.out - Y_test)^2) )
#out of sample R2
SS.total =  sum((Y_test - mean(Y_train))^2)
SS.residual = sum( (Y_test - pred.out)^2 )
r2_krr.out= 1- SS.residual / SS.total

cat('out of sample rmse:\t', rmse_krr.out)
cat('\nout of sample R2:\t', r2_krr.out,'\n')
}

## ------------------------------------------------------------------------
############ part d :  ####################
# linear regession with zipcode factor
########################################
# add zipcode to train data

variable_list = c('price','bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot')
train_data_new <- data.frame( train_data[, variable_list])
test_data_new <- data.frame( test_data[, variable_list])

# construct new training and testing data set (with interaction)
col_interact = c('bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot')
col_names <- colnames(train_data_new)

for (i in 1:3)
{
  for (j in (i+1):4)
  {
    name= col_interact[i]
    name2= col_interact[j]
    col_name = paste0(name,' * ',name2)
    #print(col_name)
      
    train_data_new[, col_name]= as.vector(as.numeric(train_data[,name])
                                         * as.numeric(train_data[,name2]))
    test_data_new[, col_name]= as.vector(as.numeric(test_data[,name])*
                                           as.numeric(test_data[,name2]))
  } 
}


train_data_new[,'zipcode'] <- train_data$zipcode
test_data_new[,'zipcode']<- test_data$zipcode


# regression
fit.lm3 <- lm(price~bedrooms + bathrooms+ sqft_living+sqft_lot+zipcode,  
              data = train_data_new)
cat('in sample R2', summary(fit.lm3)$r.squared, '\n')

fit.lm3.pred.out <- predict(fit.lm3, newdata = test_data_new)

# out-of-sample R2
SS.total <- sum((test_data$price - mean(train_data_new$price))^2)
SS.residual <- sum( (test_data$price - fit.lm3.pred.out)^2)
lm3.r2.out <- 1 - SS.residual / SS.total

cat('out of sample R2', lm3.r2.out)

## ------------------------------------------------------------------------

########### part e ##################
# spline regression 
###################################

# adding columns to the new training data set
knots <- quantile(train_data$sqft_living, 
                  p = seq(from=0.1, to=0.9, length.out = 9))
cat('quatiles \t', knots)

train_data_new['X_12'] = as.numeric(train_data$view == 0)
train_data_new['X_13'] = train_data$sqft_living**2
for (i in 1:9)
{
  str=paste0('X_',i)
  train_data_new[str]= as.numeric ( (train_data$sqft_living - knots[i])^2 * 
                                    ((train_data$sqft_living -knots[i]) >=0) )
}

cat('\ndimension of training data:', dim(train_data_new))

# create testing data set
knots_test <- quantile(test_data$sqft_living, 
                       p = seq(from=0.1, to=0.9, length.out = 9))

test_data_new['X_12'] = as.numeric(test_data$view ==0)
test_data_new['X_13'] = test_data$sqft_living**2
for (i in 1:9)
{
  str = paste0('X_',i)
  test_data_new[str] = as.numeric( (test_data$sqft_living -   knots_test[i])**2 *     
                                  ((test_data$sqft_living - knots_test[i]) >=0 ))
}

cat('\ndimension of testing data:', dim(test_data_new))



# standardize training and testing data
#zipcode_idx = grep("zipcode", colnames(train_data_new))
#view_idx = grep("X_12", colnames(train_data_new))
#for (i in 1:ncol(train_data_new))
#{
#  if ((i != zipcode_idx) && (i != view_idx))
#  {
#    train_data_new[,i] = data.matrix( scale(train_data_new[,i], 
#                                            center = TRUE, scale = TRUE) )
#    test_data_new[,i] = data.matrix( scale( test_data_new[,i], 
#                                            center = TRUE, scale = TRUE))
#  }
#}


## ------------------------------------------------------------------------
######################
# spline regression
######################
formula = as.formula(paste0("price~",
                     paste0(names(train_data_new[,2:ncol(train_data_new)]),
                            collapse="+")))

fit.lm4 <- lm(formula = formula, data = train_data_new)
cat('\n\nin sample R2\t', summary(fit.lm4)$r.squared)

fit.lm4.pred.out <- predict(fit.lm4, newdata = test_data_new)

# out-of-sample R2
SS.total <- sum((test_data_new$price - mean(train_data_new$price))^2)
SS.residual <- sum( (test_data_new$price - fit.lm4.pred.out)^2)
lm4.r2.out <- 1 - SS.residual / SS.total

cat('\nout of sample R2\t',lm4.r2.out)


## ----eval=FALSE----------------------------------------------------------
## library(knitr)
## purl(input='Zillowproblem_solution.Rmd', output='Zillowproblem_solution.R')

