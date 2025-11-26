#ifndef VGG16_HPP
#define VGG16_HPP

#include "mnist_data.hpp"
#include "convolve_layer.hpp"
#include "pool_layer.hpp"
#include "full_connect_layer.hpp"

namespace chr {
	class vgg16 {
	private:
		// 网络层定义(VGG16结构)
		// 块 1
		convolve_layer conv1_;
		convolve_layer conv2_;
		pool_layer pool1_;
		// 块 2
		convolve_layer conv3_;
		convolve_layer conv4_;
		pool_layer pool2_;
		// 块 3
		convolve_layer conv5_;
		convolve_layer conv6_;
		convolve_layer conv7_;
		pool_layer pool3_;
		// 块 4
		convolve_layer conv8_;
		convolve_layer conv9_;
		convolve_layer conv10_;
		pool_layer pool4_;
		// 块 5
		convolve_layer conv11_;
		convolve_layer conv12_;
		convolve_layer conv13_;
		pool_layer pool5_;
		// 分类器
		full_connect_layer fc1_;
		full_connect_layer fc2_;
		full_connect_layer fc3_;
	public:
		vgg16();
		Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input);
		std::vector<Eigen::MatrixXd> backward(size_t label, double learning_rate);
		double train(const std::vector<mnist_data>& dataset, size_t epochs, double learning_rate, bool show_detail = 0);
		size_t predict(const Eigen::VectorXd& output); // 预测函数
		void save(const std::filesystem::path& path);
		void load(const std::filesystem::path& path);
	private:
		Eigen::VectorXd flatten(const std::vector<Eigen::MatrixXd>& matrixs); // 展平操作(多维转一维)
		std::vector<Eigen::MatrixXd> counterflatten(const Eigen::VectorXd& vector, size_t channels, size_t rows, size_t cols); // 反展平操作(一维转多维)
		double cross_entropy_loss(const Eigen::VectorXd& output, size_t label); // 交叉熵损失函数
	};
}

#endif // !VGG16_HPP