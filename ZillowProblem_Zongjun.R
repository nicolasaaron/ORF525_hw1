
filename <- 'train.data.csv'
train_data <- read.csv(filename, header=TRUE)

filename <- 'test.data.csv'
test_data <- read.csv(filename, header=TRUE)


str(train_data)
train_data$zipcode <- as.factor(train_data$zipcode)
test_data$zipcode <- as.factor(test_data$zipcode)

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
summary(fit.lm2)$r.squared

# prediction 
test.lm2.pred <- predict(fit.lm2, newdata = test_data)

# out-of-sample R2
SS.total <- sum((test_data$price - mean(test_data$price))^2)
SS.residual <- sum( (test_data$price - test.lm2.pred)^2)
test.lm2.rsq <- 1 - SS.residual / SS.total
test.lm2.rsq



########## 1.3 kernel ridge regression with cross validation ###############
#install.packages("KRLS")
library(KRLS)

variable_list = c('bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot')

lambda = 0.1 # an arbitary lambda is chosen
gamma= 1/4
n_fold = 5

# kernel regression bandwidth parameter
# see documentation https://cran.r-project.org/web/packages/KRLS/KRLS.pdf
sigma = 1/gamma 

# if debug_flag is true, use small sample size (n<=1000) for kernel regression
# it takes about several hours to run the full sample, for small sample size, it takes around 1 mins
debug_flag = TRUE

# equally split the data into n_folds, shuffle the fold_index
fold_idx = sample(rep(1:n_fold, length.out = nrow(train_data)))


cv.mse_pred<- rep(0,n_fold)
cv.R2_out_of_sample <- rep(0,n_fold)

for (k in 1:n_fold)
{ 
  # extract subsamples
  current_fold = which(fold_idx == k)
  subsample_train = train_data[ -current_fold,]
  subsample_test = train_data[ current_fold,]
  
  # Kernel ridge regression on subsample and on columns in variable_list
  Y = subsample_train$price
  X = data.matrix(subsample_train[,variable_list])
  
  if (debug_flag) n = min(nrow(X),1000)
  else n = nrow(X)
  krr = krls(X[1:n,],Y[1:n], whichkernel = "gaussian", lambda = lambda, sigma = sigma)
  
  # compute out-sample prediction
  X_test = data.matrix(subsample_test[,variable_list])
  pred.out = predict(krr, newdata = X_test)
  
  # compute mean squared prediction error
  Y_test = subsample_test$price
  cv.mse_pred[k] =mean( (pred.out$fit - Y_test)^2 )
  
  # compute out of sample R2
  SS.total =  sum((Y_test - mean(Y_test))^2)
  SS.residual = sum( (Y_test - pred.out$fit)^2 )
  cv.R2_out_of_sample[k]= 1 - SS.residual/ SS.total
}

cv.R2_out_of_sample
cv.mse_pred

# average out of sample R2
mean(cv.R2_out_of_sample)
# average out of sample CV prediction error
mean(cv.mse_pred)





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




########## part e: spline regression ###########





