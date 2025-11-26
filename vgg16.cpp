#include "vgg16.hpp"

namespace chr {
	vgg16::vgg16()
		: conv1_(1, 3, 2, 3, 1, activation_function_type::lrelu), // 32*32
		conv2_(2, 3, 2, 1, 1, activation_function_type::lrelu),
		pool1_(2, 2), // 16*16
		conv3_(2, 3, 4, 1, 1, activation_function_type::lrelu),
		conv4_(4, 3, 4, 1, 1, activation_function_type::lrelu),
		pool2_(2, 2), // 8*8
		conv5_(4, 3, 8, 1, 1, activation_function_type::lrelu),
		conv6_(8, 3, 8, 1, 1, activation_function_type::lrelu),
		conv7_(8, 3, 8, 1, 1, activation_function_type::lrelu),
		pool3_(2, 2), // 4*4
		conv8_(8, 3, 16, 1, 1, activation_function_type::lrelu),
		conv9_(16, 3, 16, 1, 1, activation_function_type::lrelu),
		conv10_(16, 3, 16, 1, 1, activation_function_type::lrelu),
		pool4_(2, 2), // 2*2
		conv11_(16, 3, 32, 1, 1, activation_function_type::lrelu),
		conv12_(32, 3, 32, 1, 1, activation_function_type::lrelu),
		conv13_(32, 3, 32, 1, 1, activation_function_type::lrelu),
		pool5_(2, 2), // 1*1
		fc1_(32, 24, activation_function_type::lrelu),
		fc2_(24, 16, activation_function_type::lrelu),
		fc3_(16, 10, activation_function_type::lrelu) {
	}
	Eigen::VectorXd vgg16::forward(const std::vector<Eigen::MatrixXd>& input) {
		auto a1 = conv1_.forward(input);
		auto a2 = conv2_.forward(a1);
		auto p1 = pool1_.forward(a2);
		auto a3 = conv3_.forward(p1);
		auto a4 = conv4_.forward(a3);
		auto p2 = pool2_.forward(a4);
		auto a5 = conv5_.forward(p2);
		auto a6 = conv6_.forward(a5);
		auto a7 = conv7_.forward(a6);
		auto p3 = pool3_.forward(a7);
		auto a8 = conv8_.forward(p3);
		auto a9 = conv9_.forward(a8);
		auto a10 = conv10_.forward(a9);
		auto p4 = pool4_.forward(a10);
		auto a11 = conv11_.forward(p4);
		auto a12 = conv12_.forward(a11);
		auto a13 = conv13_.forward(a12);
		auto p5 = pool5_.forward(a13);
		Eigen::VectorXd f = flatten(p5);
		auto a14 = fc1_.forward(f);
		auto a15 = fc2_.forward(a14);
		return fc3_.forward(a15);
	}
	std::vector<Eigen::MatrixXd> vgg16::backward(size_t label, double learning_rate) {
		// 按照前向传播的逆序进行反向传播
		auto da15 = fc3_.backward({}, learning_rate, 1, label);
		auto da14 = fc2_.backward(da15, learning_rate);
		auto df = fc1_.backward(da14, learning_rate);
		// 将全连接层的梯度反平铺为卷积层的形状 (32个1x1的特征图)
		auto dp5 = counterflatten(df, 32, 1, 1);
		auto da13 = pool5_.backward(dp5);
		auto da12 = conv13_.backward(da13, learning_rate, 1);
		auto da11 = conv12_.backward(da12, learning_rate);
		auto da10 = conv11_.backward(da11, learning_rate);
		auto dp4 = pool4_.backward(da10);
		auto da9 = conv10_.backward(dp4, learning_rate);
		auto da8 = conv9_.backward(da9, learning_rate);
		auto da7 = conv8_.backward(da8, learning_rate);
		auto dp3 = pool3_.backward(da7);
		auto da6 = conv7_.backward(dp3, learning_rate);
		auto da5 = conv6_.backward(da6, learning_rate);
		auto da4 = conv5_.backward(da5, learning_rate);
		auto dp2 = pool2_.backward(da4);
		auto da3 = conv4_.backward(dp2, learning_rate);
		auto da2 = conv3_.backward(da3, learning_rate);
		auto dp1 = pool1_.backward(da2);
		auto da1 = conv2_.backward(dp1, learning_rate);
		return conv1_.backward(da1, learning_rate);
	}
	double vgg16::train(const std::vector<mnist_data>& dataset, size_t epochs, double learning_rate, bool show_detail) {
		double sum_accuracy = 0.0;
		if (show_detail) {
			// 显示当前数据集信息
			std::cout << "Current Dataset：";
			for (size_t i = 0; i < dataset.size(); i++) {
				std::cout << std::to_string(dataset[i].label());
				if (i >= 30) {
					std::cout << "...";
					break;
				}
			}
			std::cout << " (" << dataset.size() << "p)" << std::endl;
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
						double current_accuracy = static_cast<double>(correct) / (i + 1) * 100.0;
						// 显示进度条
						std::cout << "\rEpoch-" << epoch + 1
							<< ", Loss=" << std::fixed << std::setprecision(4) << current_loss
							<< ", Accuracy=" << std::fixed << std::setprecision(2) << current_accuracy << "%"
							<< " (" << correct << "/" << i + 1 << ") [";
						int pos = static_cast<int>(20 * progress / 100.0);
						for (int j = 0; j < 20; ++j) {
							if (j < pos) std::cout << "=";
							else std::cout << " ";
						}
						std::cout << "] " << std::setw(3) << static_cast<int>(progress) << "%    " << std::flush;
					}
				}
				// 计算epoch统计信息
				double avg_loss = loss / dataset.size();
				double accuracy = static_cast<double>(correct) / dataset.size() * 100.0;
				// 输出epoch结果
				std::cout << "\rEpoch-" << epoch + 1
					<< ", Loss=" << std::fixed << std::setprecision(4) << avg_loss
					<< ", Accuracy=" << std::fixed << std::setprecision(2) << accuracy << "%"
					<< " (" << correct << "/" << dataset.size() << ")" << std::endl;
				sum_accuracy += accuracy;
			}
		}
		else {
			for (size_t epoch = 0; epoch < epochs; ++epoch) {
				double loss = 0.0;
				size_t correct = 0;
				std::cout << "Epoch-" << epoch + 1 << " ...";
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
				std::cout << "\b\b\b\b"
					<< ", Loss=" << std::fixed << std::setprecision(4) << avg_loss
					<< ", Accuracy=" << std::fixed << std::setprecision(2) << accuracy << "%"
					<< " (" << correct << "/" << dataset.size() << ")" << std::endl;
				sum_accuracy += accuracy;
			}
		}
		return sum_accuracy /= epochs; // 返回平均准确率
	}
	size_t vgg16::predict(const Eigen::VectorXd& output) {
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
	void vgg16::save(const std::filesystem::path& path) {
		std::filesystem::create_directory(path);
		conv1_.save(path / "conv1");
		conv2_.save(path / "conv2");
		conv3_.save(path / "conv3");
		conv4_.save(path / "conv4");
		conv5_.save(path / "conv5");
		conv6_.save(path / "conv6");
		conv7_.save(path / "conv7");
		conv8_.save(path / "conv8");
		conv9_.save(path / "conv9");
		conv10_.save(path / "conv10");
		conv11_.save(path / "conv11");
		conv12_.save(path / "conv12");
		conv13_.save(path / "conv13");
		fc1_.save(path / "fc1.txt");
		fc2_.save(path / "fc2.txt");
		fc3_.save(path / "fc3.txt");
	}
	void vgg16::load(const std::filesystem::path& path) {
		std::filesystem::create_directory(path);
		conv1_.load(path / "conv1");
		conv2_.load(path / "conv2");
		conv3_.load(path / "conv3");
		conv4_.load(path / "conv4");
		conv5_.load(path / "conv5");
		conv6_.load(path / "conv6");
		conv7_.load(path / "conv7");
		conv8_.load(path / "conv8");
		conv9_.load(path / "conv9");
		conv10_.load(path / "conv10");
		conv11_.load(path / "conv11");
		conv12_.load(path / "conv12");
		conv13_.load(path / "conv13");
		fc1_.load(path / "fc1.txt");
		fc2_.load(path / "fc2.txt");
		fc3_.load(path / "fc3.txt");
	}
	Eigen::VectorXd vgg16::flatten(const std::vector<Eigen::MatrixXd>& matrixs) {
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
	std::vector<Eigen::MatrixXd> vgg16::counterflatten(const Eigen::VectorXd& vector, size_t channels, size_t rows, size_t cols) {
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
	double vgg16::cross_entropy_loss(const Eigen::VectorXd& output, size_t label) {
		// 计算softmax
		Eigen::VectorXd softmax_output = output.array().exp();
		softmax_output /= softmax_output.sum();
		// 计算交叉熵损失：-log(softmax_output[label])
		return -std::log(softmax_output[label] + 1e-8); // 添加小数值防止log(0)
	}
}