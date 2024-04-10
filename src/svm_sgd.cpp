#include <Rcpp.h> //to use the NumericVector object
#include <cmath>
#include <algorithm> // for std::random_shuffle
using namespace Rcpp; //to use the NumericVector object

#include<vector> //to use std::vector<double>

//' Stochastic Gradient Descent (SGD) algorithm for Support Vector Machines (SVM) using C++
//'
//' @param X, y of input data
//' @param epochs number of epochs
//' @param C regularization parameter
//' @param learning_rate learning rate
//' @param tolerance for stopping criterion
//' @return weight w and bias b
//' @export
// [[Rcpp::export]]
List svm_sgd_Rcpp(NumericMatrix X, NumericVector y, int epochs, double C = 1.0, double learning_rate = 1e-2, double tolerance = 1e-3) {
  int n_samples = X.nrow();
  int n_features = X.ncol();

  // Initialize weights and bias
  NumericVector w(n_features);
  double b = 0;

  // For convergence checking
  double prev_loss = R_PosInf;

  for (int epoch = 0; epoch < epochs; epoch++) {
    double total_loss = 0;

    // Creating a sequence of indices and shuffle for SGD
    IntegerVector indices = seq(0, n_samples - 1);
    std::random_shuffle(indices.begin(), indices.end());

    for (int i : indices) {
      // Random sample
      NumericVector xi = X(i, _);
      double yi = y[i];

      // Compute the decision function
      double margin = yi * (std::inner_product(xi.begin(), xi.end(), w.begin(), 0.0) + b);

      if (margin < 1) {
        // Misclassified or within the margin
        for (int j = 0; j < n_features; j++) {
          w[j] -= learning_rate * (w[j] - C * yi * xi[j]);
        }
        b -= learning_rate * (-C * yi);
        total_loss += (1 - margin);
      } else {
        // Correctly classified and outside the margin
        for (int j = 0; j < n_features; j++) {
          w[j] -= learning_rate * w[j];
        }
      }
    }

    // Calculate average loss
    double average_loss = total_loss / n_samples;

    // Check for convergence
    if (std::abs(prev_loss - average_loss) < tolerance) {
      break;
    }
    prev_loss = average_loss;

    // Learning rate decay (optional)
    learning_rate *= 0.9;
  }

  return List::create(Named("weights") = w, Named("bias") = b, Named("loss") = prev_loss);
}
