#' Stochastic Gradient Descent (SGD) algorithm for Support Vector Machines (SVM)
#'
#' @param X, y of input data
#' @param epochs number of epochs
#' @param C regularization parameter
#' @param learning_rate learning rate
#' @param tolerance for stopping criterion
#' @return weight w and bias b
svm_sgd <- function(X, y, epochs = 100, C = 1, learning_rate = 1e-2, tolerance = 1e-3) {
  n_samples <- nrow(X)
  n_features <- ncol(X)

  # Initialize weights and bias
  w <- rep(0, n_features)
  b <- 0

  # For convergence checking
  prev_loss <- Inf

  for (epoch in 1:epochs) {
    total_loss <- 0
    for (i in sample(n_samples)) {
      # Random sample
      xi <- X[i, ]
      yi <- y[i]

      # Compute the decision function
      margin <- yi * (sum(w * xi) + b)

      if (margin < 1) {
        # Misclassified or within the margin
        w <- w - learning_rate * (w - C * yi * xi)
        b <- b - learning_rate * (-C * yi)
        total_loss <- total_loss + (1 - margin)
      } else {
        # Correctly classified and outside the margin
        w <- w - learning_rate * w
      }
    }

    # Calculate average loss
    average_loss <- total_loss / n_samples

    # Check for convergence
    if (abs(prev_loss - average_loss) < tolerance) {
      break
    }
    prev_loss <- average_loss

    # Learning rate decay (optional)
    learning_rate <- learning_rate * 0.9
  }

  list(weights = w, bias = b, loss = prev_loss)
}
