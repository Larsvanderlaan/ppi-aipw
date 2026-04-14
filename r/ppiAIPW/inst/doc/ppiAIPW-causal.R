library(ppiAIPW)

set.seed(2)
n <- 120
x <- rnorm(n)
A <- sample(c("control", "treated"), n, replace = TRUE)
mu0 <- 0.5 + 0.6 * x
mu1 <- mu0 + 0.8
Y <- ifelse(A == "treated", mu1, mu0) + rnorm(n, sd = 0.4)
Yhat_potential <- cbind(control = mu0, treated = mu1)

result <- causal_inference(Y, A, Yhat_potential, method = "linear")
print(result)
summary(result)
