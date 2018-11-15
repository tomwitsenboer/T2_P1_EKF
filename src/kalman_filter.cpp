#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}
KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in; // object state
  P_ = P_in; // object covariance state
  F_ = F_in; // state transition matrix
  H_ = H_in; // measurement ,,
  R_ = R_in; // measurement covariance  ,,
  Q_ = Q_in; // Process covariance ,,
}

//Predict from lecture 5-8
void KalmanFilter::Predict() {
  x_ = F_ * x_; // u = 0
  MatrixXd Ft_ = F_.transpose();   
  P_ = F_ * P_ * Ft_ + Q_; 
}

// Generic update step KF+EKF update (lecture 5-8 + 5-21)
void KalmanFilter::KF(const VectorXd &y){
  MatrixXd Ht_ = H_.transpose();
  MatrixXd S_ = H_ * P_ * Ht_ + R_;
  MatrixXd Si_ = S_.inverse();
  MatrixXd K_ =  P_ * Ht_ * Si_;
  //new state
  int x_s = x_.size();
  MatrixXd I_ = MatrixXd::Identity(x_s, x_s); // Identity Matrix
  x_ = x_ + (K_ * y);
  P_ = (I_ - K_ * H_) * P_;
}

// Update KF part from lecture 5-8
void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
// Go to generic step KF+EKF update
  KF(y);
}

// Update EKF part  (lecture 5-21)
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd h = VectorXd(3);
  //use predicted measurement vector x_ to calculate h function
  h(0) = sqrt(x_(0)*x_(0)+x_(1)*x_(1));
  h(1) = atan2(x_(1),x_(0));
  // prevent div/o
  if (h(0))<(0.000001){
    h(0) = 0.000001;
  }
  h(2) = (x_(0)*x_(2) + x_(1) * x_(3))/h(0);
  //use h instead of H*x in K
  VectorXd y = z - h;
  //normalize angle to range -pi,pi (From: Tips and Tricks Project Description)
  y(1) = atan2(sin(y(1)),cos(y(1)));
  // Go to generic step KF + EKF update
  KF(y);
}
