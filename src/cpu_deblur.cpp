#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

static int image_type;

Mat calculatePSF() {
	int psf_size = 5;
	double mean_row = 0.0;
	double mean_col = psf_size/2.0;

	double sigma_row = 4.0;
	double sigma_col = 3.0;

	double sum = 0.0;
	double temp;
	Mat psf = Mat(Size(psf_size, psf_size), CV_64FC1, 0.0);

	for (int j = 0; j<psf.rows; j++) {
		for (int k = 0; k<psf.cols; k++) {
			temp = exp(-0.5 * (pow((j - mean_row) / sigma_row, 2.0) + pow((k - mean_col) / sigma_col, 2.0))) / (2* M_PI * sigma_row * sigma_col);
			sum += temp;
			psf.at<double>(j,k) = temp;
		}
	}

	for (int row = 0; row<psf.rows; row++) {
		for (int col = 0; col<psf.cols; col++) {
			psf.at<double>(row, col) /= sum;
		}
	}
	return psf;
}
Mat deconvlucy(Mat observed, Mat psf, int iterations) {
	Scalar initial;
	switch (image_type) {
		case CV_64FC1:
			initial = Scalar(0.5);
		case CV_64FC3:
			initial = Scalar(0.5, 0.5, 0.5);
	}
	Mat latent_est = Mat(observed.size(), image_type, initial);

	Mat psf_hat = Mat(psf.size(), CV_64FC1);
	int psf_row_max = psf.rows - 1;
	int psf_col_max = psf.cols - 1;
	for (int row = 0; row <= psf_row_max; row++) {
		for (int col = 0; col <= psf_col_max; col++) {
			psf_hat.at<double>(psf_row_max - row, psf_col_max - col) =
				psf.at<double>(row, col);
		}
	}

	Mat est_conv;
	Mat relative_blur;
	Mat error_est;

	for (int i=0; i<iterations; i++) {
		filter2D(latent_est, est_conv, -1, psf);
		relative_blur = observed.mul(1.0/est_conv);

		filter2D(relative_blur, error_est, -1, psf_hat);
		latent_est = latent_est.mul(error_est);
	}

	return latent_est;
}

int main(int argc, const char** argv)
{
	if (argc != 3) {
		cout << "Usage: " << argv[0] << " [image] [image iterations]" << endl;
		return -1;
	}
	int iterations = atoi(argv[2]);

	Mat original_img;
	original_img = imread(argv[1]);
	int channels = original_img.channels();
	switch (channels) {
		case 1:
			image_type = CV_64FC1;
			break;
		case 3:
			image_type = CV_64FC3;
			break;
		default:
			return -2;
	}

	int norm;
	switch (original_img.elemSize() / channels) {
		case 1:
			norm = 255;
			break;
		case 2:
			norm = 65535;
			break;
		default:
			return -3;
	}
	// Convet image to float matrix
	Mat float_img;
	original_img.convertTo(float_img, image_type);
	float_img *= 1.0/norm;
	namedWindow("Float", CV_WINDOW_AUTOSIZE);
	imshow("Float", float_img);

	Mat psf = calculatePSF();

	cout << "psf = " << endl << psf << endl;

	Mat estimation = deconvlucy(float_img, psf, iterations);
	namedWindow("Estimation", CV_WINDOW_AUTOSIZE);
	imshow("Estimation", estimation);

	waitKey(0);
	return 0;
}
