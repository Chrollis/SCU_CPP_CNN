#ifndef IMAGE_PROCESS_HPP
#define IMAGE_PROCESS_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <graphics.h>

namespace chr {
	namespace image_process {
		cv::Mat matrix_to_image(const Eigen::MatrixXd& matrix);
		cv::Mat easyx_to_image(const IMAGE& src);
		IMAGE image_to_easyx(const cv::Mat& src);
		Eigen::MatrixXd process_digit(const cv::Mat& digit_mat);
		void apply_padding(cv::Mat& img);
		bool is_valid_digit_region(const cv::Rect& rect, const cv::Size& image_size);
		struct digit_block {
			cv::Rect rect;
			Eigen::MatrixXd data;
		};
		std::vector<digit_block> process_image(const cv::Mat& digits_mat);
		cv::Mat labelize_image(const cv::Mat& src, const std::vector<digit_block>& blocks);
		cv::Mat binarize_image(const cv::Mat& src_img);
	}
}

#endif // !IMAGE_PROCESS_HPP
