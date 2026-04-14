library(ppiAIPW)

set.seed(3)
Y <- rbinom(100, 1, 0.5)
Yhat <- pmin(pmax(0.15 + 0.7 * Y + runif(100, -0.2, 0.2), 0.01), 0.99)
Yhat_unlabeled <- runif(300, 0.01, 0.99)

result <- mean_inference(Y, Yhat, Yhat_unlabeled, method = "linear")
diag_obj <- calibration_diagnostics(result, Y, Yhat)
plot_calibration(diag_obj)
