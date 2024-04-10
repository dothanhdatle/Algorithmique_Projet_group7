#' Sequential minimal optimization (SMO) algorithm for Support Vector Machines (SVM)
#'
#' @param X, y of input data
#' @param C regularization parameter
#' @param tol tolerance for stopping criterion
#' @param max_passes maximum number of passes to iterate over α’s without changing
#' @return Lagrange multipliers alpha and b
linear_kernel <- function(X1, X2) {
  return (X1 %*% t(X2))
}

svm_smo <- function(X, y, C = 1, tol = 1e-3, max_passes = 10, kernel_fun = linear_kernel) {
  n <- nrow(X)
  alpha <- rep(0, n)
  b <- 0
  passes <- 0
  K <- kernel_fun(X, X)

  while (passes < max_passes) {
    num_changed_alphas <- 0
    for (i in 1:n) {
      Ei <- b + sum(alpha * y * K[, i]) - y[i]
      if ((y[i] * Ei < -tol && alpha[i] < C) || (y[i] * Ei > tol && alpha[i] > 0)) {
        j <- sample(setdiff(1:n, i), 1)
        Ej <- b + sum(alpha * y * K[, j]) - y[j]

        alpha_old_i <- alpha[i]
        alpha_old_j <- alpha[j]

        L <- ifelse(y[i] != y[j], max(0, alpha[j] - alpha[i]), max(0, alpha[i] + alpha[j] - C))
        H <- ifelse(y[i] != y[j], min(C, C + alpha[j] - alpha[i]), min(C, alpha[i] + alpha[j]))

        if (L == H) next

        eta <- 2 * K[i, j] - K[i, i] - K[j, j]
        if (eta >= 0) next

        alpha[j] <- alpha[j] - y[j] * (Ei - Ej) / eta
        alpha[j] <- ifelse(alpha[j] > H, H, ifelse(alpha[j] < L, L, alpha[j]))

        if (abs(alpha[j] - alpha_old_j) < 1e-5) {
          alpha[j] <- alpha_old_j
          next
        }

        alpha[i] <- alpha[i] + y[i] * y[j] * (alpha_old_j - alpha[j])

        b1 <- b - Ei - y[i] * (alpha[i] - alpha_old_i) * K[i, i] - y[j] * (alpha[j] - alpha_old_j) * K[i, j]
        b2 <- b - Ej - y[i] * (alpha[i] - alpha_old_i) * K[i, j] - y[j] * (alpha[j] - alpha_old_j) * K[j, j]

        if (alpha[i] > 0 && alpha[i] < C) {
          b <- b1
        } else if (alpha[j] > 0 && alpha[j] < C) {
          b <- b2
        } else {
          b <- (b1 + b2) / 2
        }

        num_changed_alphas <- num_changed_alphas + 1
      }
    }
    if (num_changed_alphas == 0) {
      passes <- passes + 1
    } else {
      passes <- 0
    }
  }

  return (list(alpha = alpha, b = b))
}
