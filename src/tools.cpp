#include <iostream>
#include "tools.h"
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if (estimations.size() == 0) {
    cout << "CalculateRMSE() - Error - the estimation vector size should not be zero" << endl;
    return rmse;
  };
  if (estimations.size() != ground_truth.size()) {
    cout << "CalculateRMSE() - Error - the estimation vector size should equal ground truth vector size" << endl;
    return rmse;
  };

  for(int i=0; i < estimations.size(); ++i){
    // ... your code here
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  };

  rmse = rmse/estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3); 
  
  double c = px*px + py*py;
  double c2 = sqrt(c);
  float zero_div_th = 0.95;

  //check division by zero
  if (c < zero_div_th) {
    c = 1;
  };

  //compute the Jacobian matrix
  Hj(0, 0) = px / c2;
  Hj(0, 1) = py / c2;
  Hj(0, 2) = 0;
  Hj(0, 3) = 0;
  Hj(1, 0) = -py / c;
  Hj(1, 1) = px / c;
  Hj(1, 2) = 0;
  Hj(1, 3) = 0;
  Hj(2, 0) = py * (vx*py - vy*px) / pow(c, 1.5);
  Hj(2, 1) = px * (vy*px - vx*py) / pow(c, 1.5);
  Hj(2, 2) = px / c2;
  Hj(2, 3) = py / c2;

  return Hj;

}
