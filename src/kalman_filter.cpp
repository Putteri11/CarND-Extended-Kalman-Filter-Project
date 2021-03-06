#include "kalman_filter.h"
#include "tools.h"
#include <math.h>
#include <iostream>

#define PI 3.14159265

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;

}

void KalmanFilter::Predict() {
  
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {
  
  // KF Measurement update step
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
   
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  //new state
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
   
  VectorXd hx = VectorXd(3);
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  double rho = sqrt(px*px+py*py);
  double phi = atan2(py, px);
  double rho_dot = (px*vx+py*vy)/rho;

  hx << rho, phi, rho_dot;

  VectorXd y = z - hx;
  while (y[1] > PI) {
    y[1] -= 2*PI;
  };
  while (y[1] < -PI) {
    y[1] += 2*PI;
  };
  MatrixXd Ht = H_.transpose(); 
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  
  if (K(0) != K(0)) {
    std::cout << "R = " << R_ << std::endl;
    std::cout << "H = " << H_ << std::endl;
    std::cout << "S = " << S << std::endl;
    std::cout << "Si = " << Si << std::endl;
    std::cout << "Ht = " << Ht << std::endl;
    std::cout << "P = " << P_ << std::endl;
    std::cout << "K = " << K << std::endl;
  }

  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;

}
