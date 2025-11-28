#ifndef IMAGE_PROCESS_HPP
#define IMAGE_PROCESS_HPP

#include <Eigen/Dense>
#include <QDateTime>
#include <QImage>
#include <opencv2/opencv.hpp>

namespace chr {
namespace image_process {
    cv::Mat eigen_matrix_to_cv_mat(const Eigen::MatrixXd& matrix);
    Eigen::MatrixXd cv_mat_to_eigen_matrix(const cv::Mat& mat);
    cv::Mat qimage_to_cv_mat(const QImage& qimage);
    QImage cv_mat_to_qimage(const cv::Mat& mat);
    Eigen::MatrixXd process_digit(const cv::Mat& digit_mat);
    void apply_padding(cv::Mat& img);
    bool is_valid_digit_region(const cv::Rect& rect, const cv::Size& image_size);
    struct digit_block {
        cv::Rect rect;
        Eigen::MatrixXd data;
        QString hash;
        digit_block(const cv::Rect& rect, const Eigen::MatrixXd data);
    };
    std::vector<digit_block> process_image(const cv::Mat& digits_mat);
    cv::Mat labelize_image(const cv::Mat& src, const std::vector<digit_block>& blocks);
    cv::Mat binarize_image(const cv::Mat& src_img);
}
}

#endif // !IMAGE_PROCESS_HPP
