#include "chr_cnn.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

static void preprocess() {
    std::cout << "正在预处理MNIST数据..." << std::endl;
    chr::train_data(60000);
}

int main() {
    try {
        unsigned short size;
        double target;
        std::cout << "输入Batch大小：";
        std::cin >> size;
        std::cout << "输入目标准确率(%)：";
        std::cin >> target;
        chr::test_data(400);
        std::cout << "正在读取标签..." << std::endl;
        auto train_labels = chr::read_labels("MNIST_train_data/label.txt", size);
        auto test_labels = chr::read_labels("MNIST_test_data/label.txt");
        chr::data_loader loader;
        std::cout << "正在加载训练数据..." << std::endl;
        auto train_batches = loader.build_batches_from_directory("MNIST_train_data", size);
        std::cout << "正在加载测试数据..." << std::endl;
        auto test_batches = loader.build_batch_from_directory("MNIST_test_data");
        if (train_batches.size() != train_labels.size()) {
            std::cout << "警告: 训练数据数量(" << train_batches.size() << ")与标签数量(" << train_labels.size() << ")不匹配" << std::endl; 
            size_t min_size = std::min(train_batches.size(), train_labels.size());
            train_batches.resize(min_size);
            train_labels.resize(min_size);
        }
        std::cout << "正在初始化LeNet-5模型..." << std::endl;
        chr::le_net5 model;
        std::cout << "正在读取LeNet-5模型..." << std::endl;
        model.load("trained_lenet5_model");
        size_t epochs = 10;
        double learning_rate = 0.001;
        std::cout << "开始训练..." << std::endl;
        std::cout << "训练样本数: " << train_batches.size() << std::endl;
        std::cout << "训练轮次: " << epochs << std::endl;
        std::cout << "学习率: " << learning_rate << std::endl;
        double accuracy = 0.0;
        do{
            static size_t k = 0;
            model.train(train_batches[k], train_labels[k], epochs, learning_rate);
            k = (++k) % train_labels.size();
            std::cout << "保存模型..." << std::endl;
            model.save("trained_lenet5_model");
        } while (accuracy <= target);
        if (!test_batches.empty() && !test_labels.empty()) {
            std::cout << "在测试集上评估..." << std::endl;
            size_t correct = 0;
            size_t test_size = std::min(test_batches.size(), test_labels.size());
            for (size_t i = 0; i < test_size; ++i) {
                auto output = model.forward(test_batches[i]);
                int predicted = model.predict(output);
                if (predicted == test_labels[i]) {
                    correct++;
                }
            }
            double accuracy = static_cast<double>(correct) / test_size;
            std::cout << "测试集准确率: " << std::to_string(accuracy * 100) << "% ("
                << correct << "/" << test_size << ")" << std::endl;
        }
        std::cout << "所有操作完成!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}