
require(tictoc)

# hw1 kernel
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








######################
# preprocessing:
######################

# construct data by subsampling
#Y = train_data$price
#X = data.matrix(train_data[,variable_list])
#if (debug_flag) {
#  set.seed(0)
#  idx = sample(1:nrow(X), size= train_size)
#idx = 1:3000
#}else idx = 1:nrow(X)

  
#eually split the data into num_parts
num_parts = 7
#idx = sample(rep(1:num_parts, length.out = nrow(train_data)))
idx = rep(1:num_parts, length.out = nrow(train_data))

# splite the data
X_train = list()
Y_train = list()
for (i in 1:num_parts)
{
  current_idx = which(idx == i)
  X_train[[i]] = data.matrix(train_data[current_idx, variable_list])
  Y_train[[i]]=  as.vector(train_data[current_idx, c('price')])
  print(dim(X_train[[i]]))
}

# standardized the varaibles
for (i in 1:num_parts)
{
  X_train[[i]] = data.matrix( scale(X_train[[i]], center = TRUE, scale = TRUE) )
  Y_train[[i]] = as.vector( scale(Y_train[[i]], center = TRUE, scale = TRUE) )
}

# testing samples
X_test = data.matrix(test_data[,variable_list])
Y_test = as.vector( test_data$price)

X_test = data.matrix(scale(X_test))
Y_test = as.vector(scale(Y_test))



##########################
# use package CVST (Error in the value)
#########################
require(CVST)



data_cvst_test =constructData(X_test, Y_test)



# regression 
gamma = 1/4
lambda = 0.1
params = list(kernel="rbfdot", sigma=gamma, lambda=lambda)


reg_model = list()
pred = list()
mse = list()
R2 = list()

for (i in 1:num_parts)
{
  data_cvst_train = constructData(X_train[[i]], Y_train[[i]] )
  print(dim(data_cvst_train$x))
  print(length(data_cvst_train$y))
  
  # kernel ridge regression
  tic('kernel regression')
  krr = constructKRRLearner()
  reg_model[[i]] = krr$learn(data_cvst_train, params)
  toc()
  
  #prediction
  pred[[i]] = krr$predict(reg_model[[i]], data_cvst_test)
  
  
  #mse
  mse[[i]] = mean((pred[[i]] - Y_test)^2)
  # compute out of sample R2
  SS.total =  sum((Y_test - mean(Y_test))^2)
  SS.residual = sum( (Y_test - pred[[i]])^2 )
  R2[[i]] = 1- SS.residual / SS.total
}

print(mse)
print(R2)




##################
# using kernel lab
###################
require(kernlab)


gamma = 1/4
lambda = 0.1

mse_manual_pred= list()
R2_manual_pred = list()
for (i in 1:num_parts)
{
print(i) 
N = nrow(X_train[[i]])
rbf = rbfdot(sigma = gamma)
my_kernel<- kernelMatrix(rbf, X_train[[i]]) 

dim(my_kernel)

ident.N <- diag(rep(1,N))
tic('inversion')
alpha_x= solve(my_kernel + lambda * ident.N)
toc()

beta = alpha_x %*% Y_train[[i]]
dim(beta)

k_xy = kernelMatrix(rbf, X_test, X_train[[i]])
dim(k_xy)

pred_test = k_xy %*% beta
dim(pred_test)

#MSE
mse_manual_pred[[i]] = mean((pred_test - Y_test)^2)
# compute out of sample R2
SS.total =  sum((Y_test - mean(Y_test))^2)
SS.residual = sum( (Y_test - pred_test)^2 )
R2_manual_pred[[i]]= 1- SS.residual / SS.total
}

mse_manual_pred
R2_manual_pred


################## ####
# using Kernlab : write in a compatible way
#####################
require(kernlab)

krr.learn <- function (data, kernel, y, lambda) {
  K <- kernelMatrix(kernel, data)
  N <- nrow(K)
  alpha <- solve(Matrix(K + diag(lambda, N))) %*% y
  return(list(data = data, kernel = kernel, alpha = alpha))
}

krr.predict <- function (new_data, krr) {
  k <- kernelMatrix(krr$kernel, new_data, krr$data)
  return(k %*% krr$alpha)
}


gamma = 1/4
lambda = 0.1
my_kernel = rbfdot(sigma = gamma)

mse_krr= list()
R2_krr = list()
for (i in 1:num_parts)
{
  print(i)
  krr = krr.learn(X_train[[i]], my_kernel, Y_train[[i]], lambda)
  pred = krr.predict(X_test, krr)
  
  #MSE
  mse_krr[[i]] = mean((pred - Y_test)^2)
  # compute out of sample R2
  SS.total =  sum((Y_test - mean(Y_test))^2)
  SS.residual = sum( (Y_test - pred)^2 )
  R2_krr[[i]]= 1- SS.residual / SS.total
}

mse_krr
R2_krr





##############################
# gaussian kernel manually
#############################

#construct data
#if (debug_flag) {
#  set.seed(0)
#  idx = sample(1:nrow(train_data), size= train_size)
#  #idx = 1:3000
#}else idx = 1:nrow(train_data)

#X_train = data.matrix(train_data[idx, variable_list])
#Y_train = train_data[idx, 4]

# construct kernel

mse_manual= list()
R2_manual = list()

for (i in 1:num_parts)
{
print(i)

N <- nrow(X_train[[i]])
kk <- tcrossprod(X_train[[i]])
dd <- diag(kk)
ident.N <- diag(rep(1,N))
print(dim(kk))

gamma = 1/4
myRBF.kernel <- exp(gamma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))
dim(myRBF.kernel)

lambda =0.1
tic('kernel regression nanually')
alpha= solve(myRBF.kernel+ lambda * ident.N)
toc()

# compute hat_beta
hat_beta = alpha %*% Y_train[[i]]


# prediction
N_test = nrow(X_test)
dd_test = matrix( diag( tcrossprod(X_test) ), N_test, N)
dd_train = t( matrix( dd, N, N_test) )
kk_test = tcrossprod(X_test, X_train[[i]])
dim(kk_test)

k_star =  exp( - gamma * ( - 2 * kk_test + dd_test + dd_train))
pred = k_star %*% hat_beta


#MSE
mse_manual[[i]] = mean((pred - Y_test)^2)
# compute out of sample R2
SS.total =  sum((Y_test - mean(Y_test))^2)
SS.residual = sum( (Y_test - pred)^2 )
R2_manual[[i]] = 1- SS.residual / SS.total

}

R2_manual

