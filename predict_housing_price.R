setwd("D:/Princeton_3rd/teaching_ORF525/hw/hw1/hw1_solution/code")
filename <- 'train.data.csv'
train_data <- read.csv(filename, header=TRUE)

filename <- 'test.data.csv'
test_data <- read.csv(filename, header=TRUE)


str(train_data)
train_data$zipcode <- as.factor(train_data$zipcode)
test_data$zipcode <- as.factor(test_data$zipcode)


# if debug_flag is true, use small sample size (n<=3000) for kernel regression
# it takes about several hours to run the full sample, for small sample size, it takes around 1 mins
debug_flag = TRUE
train_size = 3000


# dependency
variable_list = c('bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot')
#variable_list = c(5:22)



##########################
# use package CVST
#########################
## another method with package CVST

require(CVST)

# construct data
Y = train_data$price
X = data.matrix(train_data[,variable_list])
if (debug_flag) {
  set.seed(0)
  idx = sample(1:nrow(X), size= train_size)
  #idx = 1:3000
}else idx = 1:nrow(X)

data_cvst_train = constructData(X[idx,], Y[idx])
dim(data_cvst_train$x)

# kernel
gamma = 1e-9
lambda = 0.1
krr = constructKRRLearner()
params = list(kernel="rbfdot", sigma=gamma, lambda=lambda)

reg_model = krr$learn(data_cvst_train, params)

# prediction
X_test = data.matrix(test_data[,variable_list])
Y_test = test_data$price
data_cvst_test =constructData(X_test, Y_test)

pred = krr$predict(reg_model, data_cvst_test)

#mse
mean((pred - data_cvst_test$y)^2)
# compute out of sample R2
SS.total =  sum((Y_test - mean(Y_test))^2)
SS.residual = sum( (Y_test - pred)^2 )
1- SS.residual / SS.total


##############################
# gaussian kernel manually
#############################

#construct data
if (debug_flag) {
  set.seed(0)
  idx = sample(1:nrow(train_data), size= train_size)
  #idx = 1:3000
}else idx = 1:nrow(train_data)

X_train = data.matrix(train_data[idx, variable_list])
Y_train = train_data[idx, 4]

# construct kernel
N <- nrow(X_train)
kk <- tcrossprod(X_train)
dd <- diag(kk)
ident.N <- diag(rep(1,N))
dim(kk)

gamma = 1/4
myRBF.kernel <- exp(gamma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))
dim(myRBF.kernel)

lambda =0.1
alpha= solve(myRBF.kernel+ lambda * ident.N)

# compute hat_beta
hat_beta = alpha %*% Y_train


# prediction
X_test = data.matrix(test_data[,variable_list])
Y_test = test_data$price

N_test = nrow(X_test)
dd_test = matrix( diag( tcrossprod(X_test) ), N_test, N)
dd_train = t( matrix( dd, N, N_test) )
kk_test = tcrossprod(X_test, X_train)
dim(kk_test)


k_star =  exp( - gamma * ( - 2 * kk_test + dd_test + dd_train))
pred = k_star %*% hat_beta


#MSE
mean((pred - Y_test)^2)
# compute out of sample R2
SS.total =  sum((Y_test - mean(Y_test))^2)
SS.residual = sum( (Y_test - pred)^2 )
1- SS.residual / SS.total


