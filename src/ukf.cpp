#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // initial predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  // Lidar measurement noise covariance matrix
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_.fill(0.0);
  R_lidar_(0,0) = std_laspx_ * std_laspx_;
  R_lidar_(1,1) = std_laspy_ * std_laspy_;

  // Radar measurement noise covariance matrix
  R_radar_ = MatrixXd(3, 3);
  R_radar_.fill(0.0);
  R_radar_(0, 0) = std_radr_ * std_radr_;
  R_radar_(1, 1) = std_radphi_ * std_radphi_;
  R_radar_(2, 2) = std_radrd_ * std_radrd_;

  // Timestamp
  previous_timestamp_ = 0;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(n_aug_);
  weights_.fill(1 / (2 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // NIS (Normalized Innovation squared)
  NIS_lidar_ = 0.0;
  NIS_radar_ = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*****************************************************************************
  *  Ellapsed time and rest
  ****************************************************************************/
	if (!is_initialized_) // only the very first measurement
		previous_timestamp_ = meas_package.timestamp_;
	//compute the time elapsed between the current and previous measurements
	float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = meas_package.timestamp_;

	//Check if the delta time is > 3 secs and reset
	// This is needed in case of reset within the simulator
	if ((dt > 3) || (dt < 0)) {
		is_initialized_ = false;
		cout << "Re-initialize" << endl;
	}
	/*****************************************************************************
	*  Initialization
	****************************************************************************/
	if (!is_initialized_) {
		// first measurement
		double p_x;
		double p_y;

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			// Polar coordinates
			double rho = meas_package.raw_measurements_[0];
			double theta = meas_package.raw_measurements_[1];
			double vrho = meas_package.raw_measurements_[2];
			// Cartesian coordinates
			p_x = rho * cos(theta);
			p_y = rho * sin(theta);
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			p_x = meas_package.raw_measurements_[0];
			p_y = meas_package.raw_measurements_[1];
		}
		// Initialize state
		x_(0) = p_x;
		x_(1) = p_y;
		x_(2) = 1; // v
		x_(3) = 1; // phi
		x_(4) = 1 ; // phid

		// (initial) state covariance matrix
		P_ = MatrixXd(5, 5);
		P_.fill(0.0);
		P_(0, 0) = 1;
		P_(1, 1) = 1;
		P_(2, 2) = 1;
		P_(3, 3) = 1;
		P_(4, 4) = 1;

		is_initialized_ = true;
		return;
	}
    
	cout << "Unscented Kalman Filter was initialized with " << endl;
	cout << "Laser: " << use_laser_ << endl << "Radar: " << use_radar_ << endl;

	/*****************************************************************************
	*  Prediction & Update
	****************************************************************************/
	Prediction(dt);

	cout << "x_ predicted = " << x_ << endl;
	cout << "ground truth = " << meas_package.raw_measurements_ << endl;

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		if (use_radar_) {
			UpdateRadar(meas_package);
		}
		else
			cout << "Radar data skipped." << endl;
	}
	else {
		// Laser updates
		if (use_laser_) {
			UpdateLidar(meas_package);
		}
		else
			cout << "Laser data skipped." << endl;
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  
  // Generate sigma points

	//create augmented mean state
	VectorXd x_aug(n_aug_);
	x_aug.head(n_x_) = x_;
	x_aug(n_x_+1) = 0;
	x_aug(n_x_+2) = 0;

	//create augmented covariance matrix
	MatrixXd P_aug(n_aug_, n_aug_);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(5, 5) = std_a_ * std_a_;
	P_aug(6, 6) = std_yawdd_ * std_yawdd_;

	cout << "Checkpoint augmented" << endl;
	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i< n_aug_; i++)
	{
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}
	cout << "Checkpoint xsig_aug " << endl;
  // Predict sigma points after time delta_t
	for (int i = 0; i<2 * n_aug_ + 1; i++) {
		VectorXd x = Xsig_aug.col(i);
		float v = x(2);
		float phi = x(3);
		float phid = x(4);
		float nu_a = x(5);
		float nu_phidd = x(6);
		//Abbreviations
		float dt2 = 0.5*delta_t*delta_t;

		if (fabs(phid) < 0.001) {
			x(0) += v * cos(phi)*delta_t + dt2 *cos(phi)*nu_a;
			x(1) += v * sin(phi)*delta_t + dt2 *sin(phi)*nu_a;
		}
		else {
			x(0) += v / phid * (sin(phi + phid * delta_t) - sin(phi)) + dt2*cos(phi)*nu_a;
			x(1) += v / phid * (-cos(phi + phid * delta_t) + cos(phi)) + dt2*sin(phi)*nu_a;
		}
		x(2) += 0 + nu_a * delta_t;
		x(3) += phid * delta_t + dt2*nu_phidd;
		x(4) += 0 + nu_phidd * delta_t;

		Xsig_pred_.col(i) = x.head(5);
	}
	cout << "Checkpoint xsig_pred" << endl;
  // Predict mean state and state covariance matrix
	//predict state mean
	x_.fill(0.0);
	for (int i = 0; i<2 * n_aug_ + 1; i++) {
		x_ += weights_(i) * Xsig_pred_.col(i);
	}

	//predict state covariance matrix
	P_.fill(0.0);
	for (int i = 0; i<2 * n_aug_ + 1; i++) {
		P_ += weights_(i) * (Xsig_pred_.col(i) - x_) * (Xsig_pred_.col(i) - x_).transpose();
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  cout << "Nothing to do here" << endl;
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	int n_z = 3;
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  // Transform sigma points into measurement space
	for (int i = 0; i<2 * n_aug_ + 1; i++) {
		float px = Xsig_pred_(0, i);
		float py = Xsig_pred_(1, i);
		float v = Xsig_pred_(2, i);
		float phi = Xsig_pred_(3, i);

		Zsig(0, i) = sqrt(px*px + py * py);
		Zsig(1, i) = atan2(py, px);
		Zsig(2, i) = (px*v*cos(phi) + py * v*sin(phi)) / Zsig(0, i);
	}

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
		z_pred = z_pred + weights_(i) * Zsig.col(i);

	//innovation covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}
	S += R_radar_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
	  //residual
	  VectorXd z_diff = Zsig.col(i) - z_pred;
	  //angle normalization
	  while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
	  while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

	  // state difference
	  VectorXd x_diff = Xsig_pred_.col(i) - x_;
	  //angle normalization
	  while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
	  while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

	  Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // New measurement z
  //residual
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // NIS (Normalized Innovation squared)
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  std::cout << "NIS (radar) = " << NIS_radar_ << endl;
}
