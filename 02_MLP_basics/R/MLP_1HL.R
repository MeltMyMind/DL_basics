library(data.table)
rm(list=ls())

# Data --------------------------------------------------------------------
dt <- fread("~/Dropbox/--- Deep Learning/# Lecture slides/# Data/Data_normalized.csv")
dt[, V1 := NULL]
dt

y <- as.matrix(dt[, Y])
X <- as.matrix(dt[, list(X1, X2)])
X <- round(X, 2)
# Weights -----------------------------------------------------------------
# IN -> HL
w1 <- matrix(data=c(-0.2, -0.4, 0.4, -0.7, -1.0, -0.8), nrow=2)
w1

# HL -> ON
w2 <- matrix(data=c(-0.6, -0.6, -0.3), nrow=3)
w2

# Biases ------------------------------------------------------------------
# HL
b1 <- 0.1  # Bias of H1
b2 <- -0.2  # Bias of H2
b3 <- 0.4  # Bias of H3
b_HL <- matrix(data=c(b1, b2, b3), nrow=1)  # Matrix of bias
B_HL <- t(matrix(b_HL, nrow=3, ncol=10)) # Stacked matix of bias, dim = N

# ON
bon <- -0.6  # Bias of ON

# Activation function -----------------------------------------------------
f_sigmoid <- function(x) 1 / (1 + exp(-x))  # Non-linear activation function
f_sigmoid_deriv <- function(x) f_sigmoid(x) * (1 - f_sigmoid(x))  # Derivative

# Cost function -----------------------------------------------------------
f_cost <- function(y, yhat) 0.5 * sum((y - yhat)^2)  # Quadratic cost function

# This cost function is used so that the "error" term in the 
# backpropagation error is the delta

# Forward pass ============================================================
# IN -> HL ----------------------------------------------------------------
A1 <- X %*% w1 + B_HL  # Sum of inputs
A1
Z1 <- f_sigmoid(A1)  # Activated
Z1
round(Z1, 1)  # For slides

# HL -> ON ----------------------------------------------------------------
A2 <- Z1 %*% w2 + bon  # Sum of inputs
A2
y_hat <- f_sigmoid(A2)  # Activated
y_hat
round(y_hat, 2)  # For slides

# Cost --------------------------------------------------------------------
cost <- f_cost(y, y_hat)  # Cost of each observation
cost
round(cost, 2)  # For slides

# Backpropagation =========================================================
alpha <- 0.5  # Learning rate

# ON -> HL
error <- -(y - y_hat)
bpe_ON <- error * f_sigmoid_deriv(A2)
BPE_ON <- cbind(bpe_ON, bpe_ON, bpe_ON)

w2_new <- w2 - alpha * BPE_ON[1,] * Z1[1,]
bon_new <- bon - alpha * BPE_ON

# HL -> IN

# Full code ===============================================================

for (iter in 1:10000){
  # Forwardpass -----------------------------------------------------------
  Z1 <- 1 / (1 + exp(-(X %*% w1 + B_HL)))
  y_hat <- 1 / (1 + exp(-(Z1 %*% w2 + bon)))
  
  # Evaluation
  ON_error <- -(y - y_hat)
  
  # Backpropagation -------------------------------------------------------
  
  # The second term should be fine because 
  # f(x) = 1 / (1 + exp(-x))
  # f'(x) f(x) (1 - f(x))
  # y_hat = f(x) and we are using the (psuedoish) derivative x * (1 - x) where x = f(x)
  bpe_ON <- ON_error * (y_hat * (1 - y_hat))
  
  HL_error <- matrix(bpe_ON, nrow=10, ncol=3) * t(matrix(w2, nrow=3, ncol=10))
  bpe_HL <- HL_error * (Z1 * (1 - Z1))
  
  w2 <- w2 - alpha * t(Z1) %*% bpe_ON  # Update w2
  w1 <- w1 - alpha * rbind(X[,1] %*% bpe_HL, X[,2] %*% bpe_HL)  # Update w1
}

