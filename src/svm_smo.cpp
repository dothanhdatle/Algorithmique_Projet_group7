#include <Rcpp.h> //to use the NumericVector object
#include <cmath>
#include <algorithm> // for std::shuffle
using namespace Rcpp; //to use the NumericVector object

#include<vector> //to use std::vector<double>

//' Sequential minimal optimization (SMO) algorithm for Support Vector Machines (SVM) using C++
//'
//' @param X, y of input data
//' @param C regularization parameter
//' @param tol tolerance for stopping criterion
//' @param max_passes maximum number of passes to iterate over α’s without changing
//' @return Lagrange multipliers alpha and b
//' @export
// [[Rcpp::export]]
NumericMatrix linear_kernel(NumericMatrix X1) {
  int n1 = X1.nrow();
  NumericMatrix result(n1, n1);

  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n1; ++j) {
      double sum = 0.0;
      for (int k = 0; k < X1.ncol(); ++k) {
        sum += X1(i, k) * X1(j, k);
      }
      result(i, j) = sum;
    }
  }
  return result;
}

// [[Rcpp::export]]
List svm_smo_Rcpp(NumericMatrix X, NumericVector y, double C = 1.0, double tol = 1e-3, int max_passes = 10) {
  int n = X.nrow();
  std::vector<double> alpha(n, 0.0);
  double b = 0.0;
  int passes = 0;
  NumericMatrix K = linear_kernel(X);
  while (passes < max_passes) {
    int num_changed_alphas = 0;
    for (int i = 0; i < n; ++i) {
      double Ei = b;
      for (int j = 0; j < n; ++j) {
        Ei += alpha[j] * y[j] * K(j,i);
      }
      Ei -= y[i];

      double yiEi = y[i] * Ei;
      if ((yiEi < -tol && alpha[i] < C) || (yiEi > tol && alpha[i] > 0)) {
        int j = rand() % n;
        while (j == i) j = rand() % n;

        double Ej = b;
        for (int k = 0; k < n; ++k) {
          Ej += alpha[k] * y[k] * K(k,j);
        }
        Ej -= y[j];

        double old_alpha_i = alpha[i];
        double old_alpha_j = alpha[j];

        double L, H;
        if (y[i] != y[j]) {
          L = std::max(0.0, alpha[j] - alpha[i]);
          H = std::min(C, C + alpha[j] - alpha[i]);
        } else {
          L = std::max(0.0, alpha[i] + alpha[j] - C);
          H = std::min(C, alpha[i] + alpha[j]);
        }

        if (L == H) continue;

        double kappa = 2 * K(i,j) - K(i,i) - K(j,j);
        if (kappa >= 0) continue;

        alpha[j] -= y[j] * (Ei - Ej) / kappa;
        alpha[j] = std::min(H, std::max(L, alpha[j]));

        if (std::abs(alpha[j] - old_alpha_j) < 1e-5) continue;

        alpha[i] += y[i] * y[j] * (old_alpha_j - alpha[j]);

        double b1 = b - Ei - y[i] * (alpha[i] - old_alpha_i) * K(i,i) -
          y[j] * (alpha[j] - old_alpha_j) * K(i,j);
        double b2 = b - Ej - y[i] * (alpha[i] - old_alpha_i) * K(i,j) -
          y[j] * (alpha[j] - old_alpha_j) * K(j,j);

        if (0 < alpha[i] && alpha[i] < C) b = b1;
        else if (0 < alpha[j] && alpha[j] < C) b = b2;
        else b = (b1 + b2) / 2;

        num_changed_alphas++;
      }
    }
    if (num_changed_alphas == 0) passes++;
    else passes = 0;
  }

  return List::create(Named("alpha") = alpha, Named("b") = b);
}
