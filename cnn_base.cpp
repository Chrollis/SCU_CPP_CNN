#include "cnn_base.h"

namespace chr {
double cnn_base::train(const std::vector<mnist_data>& dataset, size_t epochs, double learning_rate, bool show_detail)
{
    double sum_accuracy = 0.0;
    std::ostringstream oss;
    if (show_detail) {
        // 显示当前数据集信息
        oss << "Current Dataset: ";
        for (size_t i = 0; i < dataset.size(); i++) {
            oss << std::to_string(dataset[i].label());
            if (i >= 30) {
                oss << "...";
                break;
            }
        }
        oss << " (" << dataset.size() << "samples)";
        emit inform(oss.str());
        oss.clear();
        oss.str("");
        // 训练循环
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            double loss = 0.0;
            size_t correct = 0;
            for (size_t i = 0; i < dataset.size(); ++i) {
                Eigen::VectorXd output = forward({ dataset[i].image() });
                size_t predicted = predict(output);
                double sample_loss = cross_entropy_loss(output, dataset[i].label());
                if (predicted == dataset[i].label()) {
                    correct++;
                }
                loss += sample_loss;
                backward(dataset[i].label(), learning_rate);
                // 进度显示
                if (i == 0 || (i + 1) % 10 == 0 || (i + 1) == dataset.size()) {
                    double progress = static_cast<double>(i + 1) / dataset.size() * 100.0;
                    double current_loss = loss / (i + 1);
                    // 显示进度
                    emit train_details(progress, current_loss, correct, i + 1);
                }
            }
            // 计算epoch统计信息
            double avg_loss = loss / dataset.size();
            double accuracy = static_cast<double>(correct) / dataset.size() * 100.0;
            // 输出epoch结果
            oss << "Epoch " << epoch + 1
                << " - Loss: " << std::fixed << std::setprecision(4) << avg_loss
                << ", Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%"
                << " (" << correct << "/" << dataset.size() << ")";
            emit inform(oss.str());
            oss.clear();
            oss.str("");
            sum_accuracy += accuracy;
        }
    } else {
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            double loss = 0.0;
            size_t correct = 0;
            for (size_t i = 0; i < dataset.size(); ++i) {
                Eigen::VectorXd output = forward({ dataset[i].image() });
                size_t predicted = predict(output);
                double sample_loss = cross_entropy_loss(output, dataset[i].label());
                if (predicted == dataset[i].label()) {
                    correct++;
                }
                loss += sample_loss;
                backward(dataset[i].label(), learning_rate);
            }
            double avg_loss = loss / dataset.size();
            double accuracy = static_cast<double>(correct) / dataset.size() * 100.0;
            // 输出epoch结果
            oss << "Epoch " << epoch + 1
                << " - Loss: " << std::fixed << std::setprecision(4) << avg_loss
                << ", Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%"
                << " (" << correct << "/" << dataset.size() << ")";
            emit inform(oss.str());
            oss.clear();
            oss.str("");
            sum_accuracy += accuracy;
        }
    }
    return sum_accuracy / epochs; // 返回平均准确率
}
size_t cnn_base::predict(const Eigen::VectorXd& output)
{
    size_t max_index = 0;
    double max_value = output[0];
    // 找到输出向量中最大值的索引
    for (size_t i = 1; i < static_cast<size_t>(output.size()); ++i) {
        if (output[i] > max_value) {
            max_value = output[i];
            max_index = i;
        }
    }
    return max_index;
}
Eigen::VectorXd cnn_base::flatten(const std::vector<Eigen::MatrixXd>& matrixs)
{
    Eigen::VectorXd result(matrixs.size() * matrixs[0].size());
    size_t index = 0;
    for (const auto& matrix : matrixs) {
        for (size_t r = 0; r < static_cast<size_t>(matrix.rows()); r++) {
            for (size_t c = 0; c < static_cast<size_t>(matrix.cols()); c++) {
                result(index++) = matrix(r, c);
            }
        }
    }
    return result;
}
std::vector<Eigen::MatrixXd> cnn_base::counterflatten(const Eigen::VectorXd& vector, size_t channels, size_t rows, size_t cols)
{
    std::vector<Eigen::MatrixXd> result;
    size_t index = 0;
    for (size_t i = 0; i < channels; ++i) {
        Eigen::MatrixXd channel(rows, cols);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                channel(r, c) = vector[index++];
            }
        }
        result.push_back(channel);
    }
    return result;
}
double cnn_base::cross_entropy_loss(const Eigen::VectorXd& output, size_t label)
{
    // 计算softmax
    Eigen::VectorXd softmax_output = output.array().exp();
    softmax_output /= softmax_output.sum();
    // 计算交叉熵损失：-log(softmax_output[label])
    return -std::log(softmax_output[label] + 1e-8); // 添加小数值防止log(0)
}
}
