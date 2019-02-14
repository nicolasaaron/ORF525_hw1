require(tictoc)


filename <- 'train.data.csv'
train_data <- read.csv(filename, header=TRUE)

filename <- 'test.data.csv'
test_data <- read.csv(filename, header=TRUE)


str(train_data)
train_data$zipcode <- as.factor(train_data$zipcode)
test_data$zipcode <- as.factor(test_data$zipcode)



######################
# preprocessing:
######################

#standardize values of different variables, except for columns : zipcodes, X, id, date, 
zipcode_idx = grep("zipcode", colnames(train_data))
for (i in 4:ncol(train_data))
{
  if (i != zipcode_idx)
  {
    train_data[,i] = data.matrix( scale(train_data[,i], center = TRUE, scale = TRUE) )
    test_data[,i] = data.matrix( scale( test_data[,i], center = TRUE, scale = TRUE))
  }
}









########### part a ######################
fit.lm1 <- lm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot, data = train_data)
summary(fit.lm1)$r.squared

test.lm1.pred <- predict(fit.lm1, newdata = test_data)

# out-of-sample R2
SS.total <- sum((test_data$price - mean(test_data$price))^2)
SS.residual <- sum( (test_data$price - test.lm1.pred)^2)
test.lm1.rsq <- 1 - SS.residual / SS.total

test.lm1.rsq


############part b##########################
fit.lm2 <- lm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot+
                      bedrooms * bathrooms + bedrooms * sqft_living + bedrooms * sqft_lot +
                      bathrooms * sqft_living + bathrooms * sqft_lot +
                      sqft_living * sqft_lot , data = train_data)
print(paste('in sample R2:', summary(fit.lm2)$r.squared))

# prediction 
test.lm2.pred <- predict(fit.lm2, newdata = test_data)

# out-of-sample R2
SS.total <- sum((test_data$price - mean(test_data$price))^2)
SS.residual <- sum( (test_data$price - test.lm2.pred)^2)
test.lm2.rsq <- 1 - SS.residual / SS.total
print(paste('out of sample R2:', test.lm2.rsq))



########## 1.3 kernel ridge regression with cross validation ###############

######################
# using Kernlab
#####################
require(kernlab)
require(Matrix)
library("bootSVD")

krr.learn <- function (data, kernel, y, lambda) {
  K <- kernelMatrix(kernel, data)
  N <- nrow(K)
  alpha <- solve(Matrix(K + diag(lambda, N))) %*% y
  
  s <- fastSVD(K)
  singular<- s$d
  return(list(data = data, kernel = kernel, alpha = alpha, singular = singular))
}

krr.predict <- function (new_data, krr) {
  k <- kernelMatrix(krr$kernel, new_data, krr$data)
  return(k %*% krr$alpha)
}


# kernel parameters
gamma = 1/4
lambda = 0.1
my_kernel = rbfdot(sigma = gamma)

# dependency variables
variable_list = c('bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot')


# construction of subsamples (for time constraints)
# 1. eually split the data into num_parts
#    for debuging, we can try to split into 7 parts,
#   kernel ridge regression takes about 5s for each subsample
subsamples <- function(dataset, num_parts, variable_list)
{
  # split the data and then shuffle
  idx = sample(rep(1:num_parts, length.out = nrow(dataset)))
  
  X = list()
  Y = list()
  for (i in 1:num_parts)
  {
    current_idx = which(idx == i)
    X[[i]] = data.matrix( dataset[current_idx, variable_list])
    Y[[i]]=  as.vector(   dataset[current_idx, c('price')])
    print(dim(X[[i]]))
  }
  return(list(X = X, Y = Y))
}

num_parts = 1
sample_list = subsamples(train_data, num_parts, variable_list)
X_train = sample_list$X
Y_train = sample_list$Y


# 3. full testing samples
X_test = data.matrix(test_data[,variable_list])
Y_test = as.vector( test_data$price)


# 4. regression on subsamples, if num_parts = 1, then regression on full sample size
mse_krr= list()
R2_krr = list()
for (i in 1:num_parts)
{
  print(i)
  tic('krr with time: ') 
  krr = krr.learn(X_train[[i]], my_kernel, Y_train[[i]], lambda)
  toc()
  pred = krr.predict(X_test, krr)
  
  #MSE
  mse_krr[[i]] = mean((pred - Y_test)^2)
  # compute out of sample R2
  SS.total =  sum((Y_test - mean(Y_test))^2)
  SS.residual = sum( (Y_test - pred)^2 )
  R2_krr[[i]]= 1- SS.residual / SS.total
}

print('mean prediction error for each subsamples: ')
mse_krr
print('out of sample R2 for each subsamples:')
R2_krr



# 5. cross validation on subsamples

cv.krr <- function(dataset, kernel, y, lambda, folds = 5)
{
  # equally split the data into n_folds, shuffle the fold_index
  fold_idx = sample(rep(1:folds, length.out = nrow(dataset)))
  
  mean_pred_error <- rep(0,folds)
  for (k in 1:folds)
  { 
    cat('fold number:', k, '\n')
    # extract subsamples
    current_fold = which(fold_idx == k)
    X_train = dataset[ -current_fold,]
    Y_train = y[-current_fold]
    
    X_test = dataset[ current_fold,]
    Y_test = y[current_fold]
    
    tic('krr with time: ') 
    krr = krr.learn(X_train, kernel, Y_train, lambda)
    toc()
    pred = krr.predict(X_test, krr)
    
    mean_pred_error[k] <- mean( (Y_test - pred)^2 )
    cat('mean predictin error on test data:', mean_pred_error[k], '\n\n')
  }
  return(list(cv=mean(mean_pred_error), mse=mean_pred_error))
}

cv_subsamples = rep(0, num_parts)
cv_result_list = list()
for (i in num_parts)
{
  cat('subsample No', i)
  cv_result = cv.krr(X_train[[i]], my_kernel, Y_train[[i]], lambda = 0.1, folds = 5)
  
  cv_result_list[[i]] = cv_result
  cv_subsamples[i] = cv_result$cv
}



############ part d : add  zipcode ####################
fit.lm3 <- lm(price ~ bedrooms + 
                      bathrooms + 
                      sqft_living + 
                      sqft_lot + 
                      zipcode, 
              data = train_data)

summary(fit.lm3)$r.squared

test.lm3.pred <- predict(fit.lm3, newdata = test_data)

# out-of-sample R2
SS.total <- sum((test_data$price - mean(test_data$price))^2)
SS.residual <- sum( (test_data$price - test.lm3.pred)^2)
test.lm3.rsq <- 1 - SS.residual / SS.total

test.lm3.rsq

############# part e: spline 



########## part e: spline regression ###########





