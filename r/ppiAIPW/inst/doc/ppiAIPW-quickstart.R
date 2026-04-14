library(ppiAIPW)

set.seed(1)
Y <- rnorm(80)
Yhat <- Y + rnorm(80, sd = 0.35)
Yhat_unlabeled <- rnorm(200)

result <- mean_inference(
  Y,
  Yhat,
  Yhat_unlabeled,
  method = "monotone_spline",
  alpha = 0.1
)

print(result)
summary(result)
