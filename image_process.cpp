#include "image_process.hpp"
#include <QCryptographicHash>
#undef min
#undef max

namespace chr {
namespace image_process {
    // 将Eigen矩阵转换为OpenCV图像(灰度图)
    cv::Mat eigen_matrix_to_cv_mat(const Eigen::MatrixXd& matrix)
    {
        int rows = static_cast<int>(matrix.rows()), cols = static_cast<int>(matrix.cols());
        cv::Mat image(rows, cols, CV_8UC1); // 创建8位无符号单通道图像
        for (int m = 0; m < rows; m++) {
            unsigned char* ptr = image.ptr<unsigned char>(m);
            for (int n = 0; n < cols; n++) {
                ptr[n] = static_cast<unsigned char>(matrix(m, n) * 255); // 将0-1值转换为0-255
            }
        }
        return image;
    }
    Eigen::MatrixXd cv_mat_to_eigen_matrix(const cv::Mat& mat)
    {
        Eigen::MatrixXd matrix(mat.rows, mat.cols);
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                matrix(i, j) = mat.at<unsigned char>(i, j) ? 1.0 : 0.0; // 二值化
            }
        }
        return matrix;
    }
    // 处理单个数字图像：填充、缩放、二值化
    Eigen::MatrixXd process_digit(const cv::Mat& digit_mat)
    {
        cv::Mat copy = digit_mat.clone();
        apply_padding(copy); // 添加填充
        cv::Mat resized;
        cv::resize(copy, resized, cv::Size(28, 28), 0, 0, cv::INTER_AREA); // 缩放到28x28
        return cv_mat_to_eigen_matrix(resized);
    }
    // 为图像添加填充，使其接近正方形并添加边界
    void apply_padding(cv::Mat& img)
    {
        int top, bottom, left, right;
        // 根据宽高比计算填充大小
        if (img.rows > img.cols) {
            left = (img.rows - img.cols) / 2;
            right = img.rows - img.cols - left;
            top = bottom = 0;
        } else {
            top = (img.cols - img.rows) / 2;
            bottom = img.cols - img.rows - top;
            left = right = 0;
        }
        // 添加填充使图像变为正方形
        cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
        // 添加额外边界
        int border_width = img.rows / 5;
        cv::copyMakeBorder(img, img, border_width, border_width, border_width, border_width, cv::BORDER_CONSTANT, cv::Scalar(0));
    }
    // 检查矩形区域是否为有效的数字区域
    bool is_valid_digit_region(const cv::Rect& rect, const cv::Size& image_size)
    {
        int min_size = static_cast<int>(std::min(image_size.width, image_size.height) * 0.1); // 最小尺寸阈值
        min_size = std::min(min_size, 28);
        return (rect.width > min_size || rect.height > min_size);
    }
    // 处理整张图像，提取所有数字区域
    std::vector<digit_block> process_image(const cv::Mat& digits_mat)
    {
        std::vector<digit_block> digits;
        cv::Mat processed = binarize_image(digits_mat); // 二值化
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        // 查找轮廓
        cv::findContours(processed, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        for (size_t i = 0; i < contours.size(); i++) {
            cv::Rect bounding_rect = cv::boundingRect(contours[i]); // 获取边界矩形
            if (is_valid_digit_region(bounding_rect, digits_mat.size())) {
                cv::Mat digit_mat = processed(bounding_rect); // 提取数字区域
                Eigen::MatrixXd digit = process_digit(digit_mat); // 处理单个数字
                if (digit.size() > 0) {
                    digits.push_back({ bounding_rect, digit });
                }
            }
        }
        if (digits.size() <= 1)
            return digits;
        // 计算高度的中值
        std::vector<int> heights;
        for (const auto& digit : digits) {
            heights.push_back(digit.rect.height);
        }
        std::sort(heights.begin(), heights.end());
        int median_height = heights[heights.size() / 2];
        // 按y坐标排序
        std::sort(digits.begin(), digits.end(), [](const digit_block& a, const digit_block& b) {
            return a.rect.y < b.rect.y;
        });
        // 分组
        std::vector<std::vector<digit_block>> rows;
        for (const auto& digit : digits) {
            if (rows.empty()) {
                rows.push_back({ digit });
            } else {
                auto& curr_row = rows.back();
                int curr_row_y = curr_row[0].rect.y;
                if (std::abs(digit.rect.y - curr_row_y) <= median_height * 0.8) {
                    curr_row.push_back(digit);
                } else {
                    rows.push_back({ digit });
                }
            }
        }
        // 对每一行按x坐标排序
        for (auto& row : rows) {
            std::sort(row.begin(), row.end(), [](const digit_block& a, const digit_block& b) {
                return a.rect.x < b.rect.x;
            });
        }
        digits.clear(); // 将行合并为一个列表
        for (const auto& row : rows) {
            for (const auto& digit : row) {
                digits.push_back(digit);
            }
        }
        return digits;
    }
    // 为图像添加标签标记（框）
    cv::Mat labelize_image(const cv::Mat& src, const std::vector<digit_block>& blocks)
    {
        cv::Mat dst = src;
        for (size_t i = 0; i < blocks.size(); i++) {
            int x = blocks[i].rect.x;
            int y = blocks[i].rect.y;
            int w = blocks[i].rect.width;
            int h = blocks[i].rect.height;
            dst.row(y).colRange(x, x + w - 1).setTo(cv::Scalar(0, 255, 0));
            dst.row(y + h - 1).colRange(x, x + w - 1).setTo(cv::Scalar(0, 255, 0));
            dst.col(x).rowRange(y, y + h - 1).setTo(cv::Scalar(0, 255, 0));
            dst.col(x + w - 1).rowRange(y, y + h - 1).setTo(cv::Scalar(0, 255, 0));
        }
        return dst;
    }
    // 图像二值化处理
    cv::Mat binarize_image(const cv::Mat& src_img)
    {
        cv::Mat gray;
        // 转换为灰度图(如果是彩色图)
        if (src_img.channels() == 3) {
            cv::cvtColor(src_img, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = src_img.clone();
        }
        cv::Mat binary;
        cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY_INV); // 反二值化(黑底白字)
        return binary;
    }
    cv::Mat qimage_to_cv_mat(const QImage& qimage)
    {
        cv::Mat mat;
        switch (qimage.format()) {
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32:
        case QImage::Format_ARGB32_Premultiplied: {
            mat = cv::Mat(qimage.height(), qimage.width(), CV_8UC4,
                const_cast<uchar*>(qimage.bits()),
                static_cast<size_t>(qimage.bytesPerLine()));
            cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
            break;
        }
        case QImage::Format_RGB888: {
            mat = cv::Mat(qimage.height(), qimage.width(), CV_8UC3,
                const_cast<uchar*>(qimage.bits()),
                static_cast<size_t>(qimage.bytesPerLine()));
            cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
            break;
        }
        case QImage::Format_Grayscale8: {
            mat = cv::Mat(qimage.height(), qimage.width(), CV_8UC1,
                const_cast<uchar*>(qimage.bits()),
                static_cast<size_t>(qimage.bytesPerLine()));
            break;
        }
        default: {
            // 如果不支持格式，先转换为 RGB888
            QImage converted = qimage.convertToFormat(QImage::Format_RGB888);
            mat = cv::Mat(converted.height(), converted.width(), CV_8UC3,
                const_cast<uchar*>(converted.bits()),
                static_cast<size_t>(converted.bytesPerLine()));
            cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
            break;
        }
        }
        return mat.clone(); // 返回深拷贝以避免内存问题
    }
    QImage cv_mat_to_qimage(const cv::Mat& mat)
    {
        switch (mat.type()) {
        case CV_8UC1: {
            QImage image(mat.data, mat.cols, mat.rows,
                static_cast<int>(mat.step), QImage::Format_Grayscale8);
            return image.copy();
        }
        case CV_8UC3: {
            QImage image(mat.data, mat.cols, mat.rows,
                static_cast<int>(mat.step), QImage::Format_RGB888);
            return image.rgbSwapped().copy(); // BGR -> RGB
        }
        case CV_8UC4: {
            QImage image(mat.data, mat.cols, mat.rows,
                static_cast<int>(mat.step), QImage::Format_ARGB32);
            return image.copy();
        }
        default: {
            // 如果不支持格式，先转换为 8UC3
            cv::Mat converted;
            mat.convertTo(converted, CV_8UC3);
            QImage image(converted.data, converted.cols, converted.rows,
                static_cast<int>(converted.step), QImage::Format_RGB888);
            return image.rgbSwapped().copy();
        }
        }
    }
    digit_block::digit_block(const cv::Rect& rect, const Eigen::MatrixXd data)
    {
        this->rect = rect;
        this->data = data;
        qint64 timestamp = QDateTime::currentMSecsSinceEpoch();
        QCryptographicHash qhash(QCryptographicHash::Sha256);
        qhash.addData(reinterpret_cast<const char*>(&timestamp));
        QByteArray bits;
        unsigned char bit = 0;
        int counter = 0;
        for (int i = 0; i < data.rows(); i++) {
            for (int j = 0; j < data.cols(); j++) {
                if (data(i, j) > 0.5) {
                    bit |= (1 << (7 - counter));
                }
                if (++counter == 8) {
                    bits.append(bit);
                    bit = 0;
                    counter = 0;
                }
            }
        }
        if (counter > 0) {
            bits.append(bit);
        }
        qhash.addData(bits);
        hash = qhash.result().toHex();
    }
}
}
