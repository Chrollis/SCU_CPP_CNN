#include "chr_cnn.hpp"

namespace chr {

	namespace preprocess {
		unsigned swap_endian(unsigned val) {
			val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
			return (val << 16) | (val >> 16);
		}
		void decompress_mnist(const std::filesystem::path& mnist_img_path, const std::filesystem::path& mnist_label_path, const std::filesystem::path& output_path) {
			std::ifstream mnist_img(mnist_img_path, std::ios::binary);
			std::ifstream mnist_label(mnist_label_path, std::ios::binary);
			if (!mnist_img.is_open()) {
				throw std::runtime_error("打开MNIST图像文件失败");
			}
			if (!mnist_label.is_open()) {
				throw std::runtime_error("打开MNIST标签文件失败");
			}
			unsigned magic = 0;
			unsigned items = 0, labels = 0;
			unsigned rows = 0, cols = 0;
			mnist_img.read(reinterpret_cast<char*>(&magic), 4);
			magic = swap_endian(magic);
			if (magic != 2051) {
				throw std::runtime_error("所选文件不是MNIST图像文件");
			}
			mnist_label.read(reinterpret_cast<char*>(&magic), 4);
			magic = swap_endian(magic);
			if (magic != 2049) {
				throw std::runtime_error("所选文件不是MNIST标签文件");
			}
			mnist_img.read(reinterpret_cast<char*>(&items), 4);
			items = swap_endian(items);
			mnist_label.read(reinterpret_cast<char*>(&labels), 4);
			labels = swap_endian(labels);
			if (items != labels) {
				throw std::runtime_error("图像文件与标签文件不成对");
			}
			mnist_img.read(reinterpret_cast<char*>(&rows), 4);
			rows = swap_endian(rows);
			mnist_img.read(reinterpret_cast<char*>(&cols), 4);
			cols = swap_endian(cols);
			std::ofstream file(output_path / "label.txt");
			std::vector<unsigned char> pixels(rows * cols);
			unsigned char label = 0;
			for (size_t i = 0; i < items; i++) {
				mnist_img.read(reinterpret_cast<char*>(&pixels), static_cast<std::streamsize>(rows) * cols);
				mnist_label.read(reinterpret_cast<char*>(&label), 1);
				file << std::to_string(label) + " ";
				cv::Mat img(rows, cols, CV_8UC1);
				for (size_t m = 0; m < rows; m++) {
					uchar* ptr = img.ptr<uchar>(static_cast<int>(m));
					for (size_t n = 0; n < cols; n++) {
						ptr[n] = static_cast<unsigned char>(pixels[m * cols + n]) / 255.0;
					}
				}
				std::string out = (output_path / (std::to_string(i) + ".jpg")).string();
				cv::imwrite(out, img);
			}
			mnist_img.close();
			mnist_label.close();
			file.close();
		}
		void decompress_data() {
			try {
				std::filesystem::create_directory("MNIST_train_data");
				decompress_mnist("MNIST_data/train-images.idx3-ubyte", "MNIST_data/train-labels.idx1-ubyte", "MNIST_train_data");
				std::filesystem::create_directory("MNIST_test_data");
				decompress_mnist("MNIST_data/t10k-images.idx3-ubyte", "MNIST_data/t10k-labels.idx1-ubyte", "MNIST_test_data");
			}
			catch (const std::exception& e) {
				std::cout << e.what() << "\n";
			}
		}
		std::vector<Eigen::MatrixXd> padding(const std::vector<Eigen::MatrixXd>& input, size_t circle_num, double fill_num) {
			std::vector<Eigen::MatrixXd> result;
			for (const auto& channel : input) {
				size_t new_rows = channel.rows() + 2 * circle_num;
				size_t new_cols = channel.cols() + 2 * circle_num;
				Eigen::MatrixXd padded = Eigen::MatrixXd::Constant(new_rows, new_cols, fill_num);
				padded.block(circle_num, circle_num, channel.rows(), channel.cols()) = channel;
				result.push_back(padded);
			}
			return result;
		}
		Eigen::MatrixXd padding(const Eigen::MatrixXd& input, size_t circle_num, double fill_num) {
			size_t new_rows = input.rows() + 2 * circle_num;
			size_t new_cols = input.cols() + 2 * circle_num;
			Eigen::MatrixXd result = Eigen::MatrixXd::Constant(new_rows, new_cols, fill_num);
			result.block(circle_num, circle_num, input.rows(), input.cols()) = input;
			return result;
		}
		cv::Mat binarize_img(const cv::Mat& src_img) {
			cv::Mat gray;
			if (src_img.channels() == 3) {
				cv::cvtColor(src_img, gray, cv::COLOR_BGR2GRAY);
			}
			else {
				gray = src_img.clone();
			}
			cv::Mat binary;
			cv::threshold(gray, binary, 150, 255, cv::THRESH_BINARY_INV);
			return binary;
		}
	}

	namespace data_loader {
		Eigen::MatrixXd process_digit(cv::Mat& digit_mat) {
			apply_padding(digit_mat);
			cv::Mat resized;
			cv::resize(digit_mat, resized, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
			Eigen::MatrixXd eigen_digit(28, 28);
			for (size_t i = 0; i < 28; i++) {
				for (size_t j = 0; j < 28; j++) {
					eigen_digit(i, j) = resized.at<uchar>(i, j) / 255.0;
				}
			}
			return eigen_digit;
		}
		void apply_padding(cv::Mat& img) {
			unsigned top, bottom, left, right;
			if (img.rows > img.cols) {
				left = (img.rows - img.cols) / 2;
				right = img.rows - img.cols - left;
				top = bottom = 0;
			}
			else {
				top = (img.cols - img.rows) / 2;
				bottom = img.cols - img.rows - top;
				left = right = 0;
			}
			cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
			unsigned border_width = img.rows / 5;
			cv::copyMakeBorder(img, img, border_width, border_width, border_width, border_width, cv::BORDER_CONSTANT, cv::Scalar(0));
		}
		bool is_valid_digit_region(const cv::Rect& rect, const cv::Size& image_size) {
			double min_ratio = 0.1;
			return (rect.width > image_size.width * min_ratio && rect.height > image_size.height * min_ratio);
		}
		Eigen::MatrixXd read_image_to_Eigen(const std::filesystem::path& path) {
			cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
			if (image.empty()) {
				return Eigen::MatrixXd();
			}
			Eigen::MatrixXd result(image.rows, image.cols);
			for (int i = 0; i < image.rows; ++i) {
				for (int j = 0; j < image.cols; ++j) {
					result(i, j) = static_cast<double>(image.at<uchar>(i, j)) / 255.0;
				}
			}
			return result;
		}
		std::vector<Eigen::MatrixXd> process_plural_digits(cv::Mat& digits_mat) {
			std::vector<Eigen::MatrixXd> digits;
			cv::Mat processed = preprocess::binarize_img(digits_mat);
			std::vector<std::vector<cv::Point>> contours;
			std::vector<cv::Vec4i> hierarchy;
			cv::findContours(processed, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
			for (size_t i = 0; i < contours.size(); i++) {
				cv::Rect bounding_rect = cv::boundingRect(contours[i]);
				if (is_valid_digit_region(bounding_rect, digits_mat.size())) {
					cv::Mat digit_mat = processed(bounding_rect);
					Eigen::MatrixXd digit = process_digit(digit_mat);
					if (digit.size() > 0) {
						digits.push_back(digit);
					}
				}
			}
			return digits;
		}
		std::vector<std::vector<short>> mnist_label_batches(const std::filesystem::path& label_file_path, size_t total_size, size_t batch_size) {
			std::vector<std::vector<short>> batches;
			std::vector<short> batch;
			short label = 0;
			std::ifstream file(label_file_path);
			for (size_t i = 0; i < total_size; i++) {
				file >> label;
				batch.push_back(label);
				if (batch.size() >= batch_size) {
					batches.push_back(std::move(batch));
					batch.clear();
				}
			}
			if (!batch.empty()) {
				batches.push_back(batch);
				batch.clear();
			}
			return batches;
		}
		std::vector<std::vector<std::vector<Eigen::MatrixXd>>> mnist_image_batches(const std::vector<std::filesystem::path>& img_paths, size_t batch_size) {
			std::vector<std::vector<std::vector<Eigen::MatrixXd>>> batches;
			std::vector<std::vector<Eigen::MatrixXd>> batch;
			for (const auto& path : img_paths) {
				std::vector<Eigen::MatrixXd> digit;
				digit.push_back(std::move(read_image_to_Eigen(path)));
				batch.push_back(digit);
				if (batch.size() >= batch_size) {
					batches.push_back(std::move(batch));
					batch.clear();
				}
			}
			if (!batch.empty()) {
				batches.push_back(batch);
				batch.clear();
			}
			return batches;
		}
		std::vector<std::vector<std::vector<Eigen::MatrixXd>>> mnist_image_batches_from_directory(const std::filesystem::path& img_dir, size_t total_size, size_t batch_size) {
			std::vector<std::filesystem::path> img_paths;
			for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
				if (entry.path().extension() == ".jpg") {
					img_paths.push_back(entry.path());
				}
				if (img_paths.size() >= total_size) break;
			}
			return mnist_image_batches(img_paths, batch_size);
		}
	}

	activation_function::activation_function() {
		function = [this](double x) {return sigmoid(x); };
		derivative = [this](double x) {return sigmoid_derivative(x); };
	}
	void activation_function::change_function(activation_function_type type) {
		switch (type)
		{
		case activation_function_type::sigmoid:
			function = [this](double x) {return sigmoid(x); };
			derivative = [this](double x) {return sigmoid_derivative(x); };
			break;
		case activation_function_type::tanh:
			function = [this](double x) {return tanh(x); };
			derivative = [this](double x) {return tanh_derivative(x); };
			break;
		case activation_function_type::relu:
			function = [this](double x) {return relu(x); };
			derivative = [this](double x) {return relu_derivative(x); };
			break;
		case activation_function_type::lrelu:
			function = [this](double x) {return lrelu(x); };
			derivative = [this](double x) {return lrelu_derivative(x); };
			break;
		default:
			break;
		}
	}

	void filter::initialize_gausz(double stddev) {
		kernels_.resize(channel_);
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::normal_distribution<double> distributor(0.0, stddev);
		bias_ = distributor(generator);
		for (size_t i = 0; i < channel_; i++) {
			kernels_[i] = Eigen::MatrixXd::Zero(width_, width_);
			for (size_t m = 0; m < width_; m++) {
				for (size_t n = 0; n < width_; n++) {
					kernels_[i](m, n) = distributor(generator);
				}
			}
		}
	}
	void filter::initialize_xavier(size_t input_size) {
		kernels_.resize(channel_);
		std::random_device rd;
		std::default_random_engine generator(rd());
		double limit = sqrt(6.0 / (input_size + channel_ * width_ * width_));
		std::uniform_real_distribution<double> distributor(-limit, limit);
		bias_ = 0;
		for (size_t i = 0; i < channel_; i++) {
			kernels_[i] = Eigen::MatrixXd::Zero(width_, width_);
			for (size_t m = 0; m < width_; m++) {
				for (size_t n = 0; n < width_; n++) {
					kernels_[i](m, n) = distributor(generator);
				}
			}
		}
	}
	void filter::initialize_He(size_t input_size) {
		kernels_.resize(channel_);
		std::random_device rd;
		std::default_random_engine generator(rd());
		double stddev = sqrt(2.0 / input_size);
		std::normal_distribution<double> distributor(0.0, stddev);
		bias_ = 0;
		for (size_t i = 0; i < channel_; i++) {
			kernels_[i] = Eigen::MatrixXd::Zero(width_, width_);
			for (size_t m = 0; m < width_; m++) {
				for (size_t n = 0; n < width_; n++) {
					kernels_[i](m, n) = distributor(generator);
				}
			}
		}
	}
	void filter::save(const std::filesystem::path& path) const {
		std::ofstream file(path);
		if (file.is_open()) {
			file << channel_ << " " << width_ << " " << bias_ << "\n";
			for (int i = 0; i < channel_; ++i) {
				for (int row = 0; row < width_; ++row) {
					for (int col = 0; col < width_; ++col) {
						file << kernels_[i](row, col) << " ";
					}
					file << "\n";
				}
				file << "\n";
			}
			file.close();
		}
	}
	void filter::load(const std::filesystem::path& path) {
		std::ifstream file(path);
		if (file.is_open()) {
			file >> channel_ >> width_ >> bias_;
			kernels_.resize(channel_);
			for (int i = 0; i < channel_; ++i) {
				kernels_[i] = Eigen::MatrixXd::Zero(width_, width_);
				for (int row = 0; row < width_; ++row) {
					for (int col = 0; col < width_; ++col) {
						file >> kernels_[i](row, col);
					}
				}
			}
			file.close();
		}
	}

	std::vector<Eigen::MatrixXd> pool_layer::forward(const std::vector<Eigen::MatrixXd>& input) {
		input_ = preprocess::padding(input, padding_, 0.0);
		feature_map_.clear();
		record_.clear();
		for (const auto& channel : input_) {
			size_t output_rows = (channel.rows() - size_) / stride_ + 1;
			size_t output_cols = (channel.cols() - size_) / stride_ + 1;
			Eigen::MatrixXd output = Eigen::MatrixXd::Zero(output_rows, output_cols);
			if (type_ == pooling_type::max) {
				record_.push_back(Eigen::MatrixXd::Zero(channel.rows(), channel.cols()));
				max_pooling(channel, output, record_.back());
			}
			else {
				average_pooling(channel, output);
			}
			feature_map_.push_back(output);
		}
		return feature_map_;
	}
	std::vector<Eigen::MatrixXd> pool_layer::backward(const std::vector<Eigen::MatrixXd>& loss) {
		std::vector<Eigen::MatrixXd> new_loss;
		for (size_t i = 0; i < loss.size(); ++i) {
			if (type_ == pooling_type::max) {
				new_loss.push_back(max_backward(loss[i], record_[i]));
			}
			else {
				new_loss.push_back(average_backward(loss[i]));
			}
		}
		return remove_padding(new_loss, padding_);
	}
	void pool_layer::max_pooling(const Eigen::MatrixXd& input, Eigen::MatrixXd& output, Eigen::MatrixXd& record) const {
		for (size_t i = 0; i < output.rows(); ++i) {
			for (size_t j = 0; j < output.cols(); ++j) {
				double max_val = std::numeric_limits<double>::lowest();
				size_t max_row = 0, max_col = 0;
				for (size_t m = 0; m < size_; ++m) {
					for (size_t n = 0; n < size_; ++n) {
						double val = input(i * stride_ + m, j * stride_ + n);
						if (val > max_val) {
							max_val = val;
							max_row = i * stride_ + m;
							max_col = j * stride_ + n;
						}
					}
				}
				output(i, j) = max_val;
				record(max_row, max_col) = 1.0;
			}
		}
	}
	void pool_layer::average_pooling(const Eigen::MatrixXd& input, Eigen::MatrixXd& output) const {
		double divisor = size_ * size_;
		for (size_t i = 0; i < output.rows(); ++i) {
			for (size_t j = 0; j < output.cols(); ++j) {
				double sum = 0.0;
				for (size_t m = 0; m < size_; ++m) {
					for (size_t n = 0; n < size_; ++n) {
						sum += input(i * stride_ + m, j * stride_ + n);
					}
				}
				output(i, j) = sum / divisor;
			}
		}
	}
	Eigen::MatrixXd pool_layer::max_backward(const Eigen::MatrixXd& loss, const Eigen::MatrixXd& record) const {
		Eigen::MatrixXd new_loss = Eigen::MatrixXd::Zero(record.rows(), record.cols());
		for (size_t i = 0; i < loss.rows(); ++i) {
			for (size_t j = 0; j < loss.cols(); ++j) {
				for (size_t m = 0; m < size_; ++m) {
					for (size_t n = 0; n < size_; ++n) {
						size_t row = i * stride_ + m;
						size_t col = j * stride_ + n;
						new_loss(row, col) += loss(i, j) * record(row, col);
					}
				}
			}
		}
		return new_loss;
	}
	Eigen::MatrixXd pool_layer::average_backward(const Eigen::MatrixXd& loss) {
		Eigen::MatrixXd new_loss = Eigen::MatrixXd::Zero(input_[0].rows(), input_[0].cols());
		double divisor = size_ * size_;
		for (size_t i = 0; i < loss.rows(); ++i) {
			for (size_t j = 0; j < loss.cols(); ++j) {
				for (size_t m = 0; m < size_; ++m) {
					for (size_t n = 0; n < size_; ++n) {
						new_loss(i * stride_ + m, j * stride_ + n) += loss(i, j) / divisor;
					}
				}
			}
		}
		return new_loss;
	}
	std::vector<Eigen::MatrixXd> pool_layer::remove_padding(const std::vector<Eigen::MatrixXd>& input, size_t padding) {
		if (padding == 0) return input;
		std::vector<Eigen::MatrixXd> result;
		for (const auto& matrix : input) {
			result.push_back(matrix.block(padding, padding,
				matrix.rows() - 2 * padding,
				matrix.cols() - 2 * padding));
		}
		return result;
	}

	convolve_layer::convolve_layer(size_t in_channel, size_t kernel_size, size_t out_channel, size_t padding, size_t stride, activation_function_type activate_type)
		: in_channel_(in_channel), kernel_size_(kernel_size), out_channel_(out_channel), padding_(padding), stride_(stride), afunc_(activate_type) {
		for (size_t i = 0; i < out_channel; ++i) {
			filters_.emplace_back(in_channel, kernel_size);
			filters_.back().initialize_He(in_channel * kernel_size * kernel_size);
		}
	}
	std::vector<Eigen::MatrixXd> convolve_layer::forward(const std::vector<Eigen::MatrixXd>& input) {
		input_ = preprocess::padding(input, padding_, 0.0);
		feature_map_.clear();
		d_feature_map_.clear();
		for (auto& filter : filters_) {
			long output_rows = (input_[0].rows() - kernel_size_) / stride_ + 1;
			long output_cols = (input_[0].cols() - kernel_size_) / stride_ + 1;
			Eigen::MatrixXd channel_result = Eigen::MatrixXd::Zero(output_rows, output_cols);
			for (size_t i = 0; i < in_channel_; ++i) {
				Eigen::MatrixXd conv_result = convolve(input_[i], filter.kernels_[i], stride_);
				channel_result += conv_result;
			}
			channel_result.array() += filter.bias_;
			d_feature_map_.push_back(channel_result.unaryExpr([this](double x) { return afunc_[x]; }));
			feature_map_.push_back(channel_result.unaryExpr([this](double x) { return afunc_(x); }));
		}
		return feature_map_;
	}
	std::vector<Eigen::MatrixXd> convolve_layer::backward(const std::vector<Eigen::MatrixXd>& loss, bool is_last_conv) {
		std::vector<Eigen::MatrixXd> propagated_loss;
		if (!is_last_conv) {
			propagated_loss = apply_activation_derivative(loss);
		}
		else {
			propagated_loss = loss;
		}
		std::vector<Eigen::MatrixXd> input_loss(in_channel_, Eigen::MatrixXd::Zero(input_[0].rows(), input_[0].cols()));
		for (size_t i = 0; i < out_channel_; ++i) {
			for (size_t j = 0; j < in_channel_; ++j) {
				Eigen::MatrixXd rotated_kernel = filters_[i].kernels_[j].reverse();
				size_t pad_size = kernel_size_ - 1;
				std::vector<Eigen::MatrixXd> single_loss_vec = { propagated_loss[i] };
				auto padded_vec = preprocess::padding(single_loss_vec, pad_size, 0.0);
				if (!padded_vec.empty() &&
					padded_vec[0].rows() >= rotated_kernel.rows() &&
					padded_vec[0].cols() >= rotated_kernel.cols()) {
					Eigen::MatrixXd conv_result = convolve(padded_vec[0], rotated_kernel, 1);
					if (conv_result.rows() == input_loss[j].rows() &&
						conv_result.cols() == input_loss[j].cols()) {
						input_loss[j] += conv_result;
					}
					else {
						// throw std::runtime_error("卷积反向传播中卷积结果维度不匹配");
					}
				}
			}
		}
		return remove_padding(input_loss, padding_);
	}
	void convolve_layer::weights_update(double learning_rate, const std::vector<Eigen::MatrixXd>& loss) {
		if (loss.size() != out_channel_) {
			// throw std::runtime_error("损失大小与输出通道数不匹配");
			return;
		}
		if (input_.size() != in_channel_) {
			// throw std::runtime_error("输入大小与输入通道数不匹配");
			return;
		}
		for (size_t i = 0; i < out_channel_; ++i) {
			if (i >= loss.size()) {
				// throw std::runtime_error("损失索引超出范围");
				continue;
			}
			for (size_t j = 0; j < in_channel_; ++j) {
				if (j >= input_.size()) {
					// throw std::runtime_error("输入索引超出范围");
					continue;
				}
				if (i >= filters_.size()) {
					// throw std::runtime_error("滤波器索引超出范围");
					continue;
				}
				if (j >= filters_[i].kernels_.size()) {
					// throw std::runtime_error("卷积核索引超出范围");
					continue;
				}
				try {
					Eigen::MatrixXd grad = compute_weight_gradient(input_[j], loss[i]);
					filters_[i].kernels_[j] -= learning_rate * grad;
				}
				catch (const std::exception& e) {
					throw std::runtime_error(std::string("计算权重梯度时出错: ") + e.what());
				}
			}

			double bias_grad = loss[i].sum();
			filters_[i].bias_ -= learning_rate * bias_grad;
		}
	}
	void convolve_layer::save(const std::filesystem::path& path) const {
		for (size_t i = 0; i < filters_.size(); ++i) {
			filters_[i].save(path / ("filter_" + std::to_string(i) + ".txt"));
		}
	}
	void convolve_layer::load(const std::filesystem::path& path) {
		for (size_t i = 0; i < filters_.size(); ++i) {
			filters_[i].load(path / ("filter_" + std::to_string(i) + ".txt"));
		}
	}
	Eigen::MatrixXd convolve_layer::convolve(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel, size_t stride) const {
		if (input.rows() < kernel.rows() || input.cols() < kernel.cols()) {
			throw std::invalid_argument("输入维度小于卷积核维度");
		}
		long output_rows = static_cast<long>((input.rows() - kernel.rows()) / stride) + 1;
		long output_cols = static_cast<long>((input.cols() - kernel.cols()) / stride) + 1;
		if (output_rows <= 0 || output_cols <= 0) {
			throw std::invalid_argument("卷积后输出维度无效");
		}
		Eigen::MatrixXd result = Eigen::MatrixXd::Zero(output_rows, output_cols);
		Eigen::Map<const Eigen::VectorXd> kernel_flat(kernel.data(), kernel.size());
		for (long i = 0; i < output_rows; i++) {
			for (long j = 0; j < output_cols; j++) {
				long start_row = i * stride;
				long start_col = j * stride;
				if (start_row + kernel.rows() <= input.rows() &&
					start_col + kernel.cols() <= input.cols()) {
					Eigen::MatrixXd block = input.block(start_row, start_col, kernel.rows(), kernel.cols());
					Eigen::Map<const Eigen::VectorXd> block_flat(block.data(), block.size());
					result(i, j) = block_flat.dot(kernel_flat);
				}
			}
		}
		return result;
	}
	std::vector<Eigen::MatrixXd> convolve_layer::apply_activation_derivative(const std::vector<Eigen::MatrixXd>& loss) {
		std::vector<Eigen::MatrixXd> result;
		for (size_t i = 0; i < loss.size(); ++i) {
			result.push_back(loss[i].cwiseProduct(d_feature_map_[i]));
		}
		return result;
	}
	std::vector<Eigen::MatrixXd> convolve_layer::remove_padding(const std::vector<Eigen::MatrixXd>& input, size_t padding) {
		if (padding == 0) return input;
		std::vector<Eigen::MatrixXd> result;
		for (const auto& matrix : input) {
			result.push_back(matrix.block(padding, padding, matrix.rows() - 2 * padding, matrix.cols() - 2 * padding));
		}
		return result;
	}

	Eigen::VectorXd full_connect_layer::forward(const Eigen::VectorXd& input) {
		input_ = input;
		feature_vector_ = weights_ * input_ + biases_;
		for (size_t i = 0; i < out_size_; ++i) {
			double pre_activation = feature_vector_(i);
			d_feature_vector_(i) = afunc_[pre_activation];
			feature_vector_(i) = afunc_(pre_activation);
		}
		return feature_vector_;
	}
	Eigen::VectorXd full_connect_layer::backward(const Eigen::VectorXd& loss, double learning_rate, bool is_output_layer, int label) {
		if (is_output_layer) {
			Eigen::VectorXd target = Eigen::VectorXd::Zero(out_size_);
			if (label >= 0 && label < out_size_) {
				target(label) = 1.0;
			}
			loss_ = feature_vector_ - target;
		}
		else {
			if (loss.size() == d_feature_vector_.size()) {
				loss_ = loss.cwiseProduct(d_feature_vector_);
			}
			else {
				// throw std::runtime_error("隐藏层损失维度不匹配");
				loss_ = Eigen::VectorXd::Zero(out_size_);
			}
		}
		weights_update(learning_rate);
		return weights_.transpose() * loss_;
	}
	void full_connect_layer::weights_update(double learning_rate) {
		double grad_norm = (loss_ * input_.transpose()).norm();
		if (grad_norm > 1.0) {
			loss_ = loss_ / grad_norm;
		}
		weights_ -= learning_rate * loss_ * input_.transpose();
		biases_ -= learning_rate * loss_;
	}
	Eigen::VectorXd full_connect_layer::get_flatten(const std::vector<Eigen::MatrixXd>& feature_map) {
		size_t total_size = 0;
		for (const auto& matrix : feature_map) {
			total_size += matrix.size();
		}
		Eigen::VectorXd flatten(total_size);
		size_t index = 0;
		for (const auto& matrix : feature_map) {
			Eigen::Map<const Eigen::VectorXd> flat(matrix.data(), matrix.size());
			flatten.segment(index, matrix.size()) = flat;
			index += matrix.size();
		}
		return flatten;
	}
	void full_connect_layer::save(const std::filesystem::path& path) const {
		std::ofstream file(path);
		if (file.is_open()) {
			file << in_size_ << " " << out_size_ << "\n";
			file << weights_ << "\n";
			file << biases_.transpose() << "\n";
			file.close();
		}
	}
	void full_connect_layer::load(const std::filesystem::path& path) {
		std::ifstream file(path);
		if (file.is_open()) {
			size_t in_size, out_size;
			file >> in_size >> out_size;
			weights_.resize(out_size, in_size);
			biases_.resize(out_size);
			for (size_t i = 0; i < out_size; ++i) {
				for (size_t j = 0; j < in_size; ++j) {
					file >> weights_(i, j);
				}
			}
			for (size_t i = 0; i < out_size; ++i) {
				file >> biases_(i);
			}
			file.close();
		}
	}
	void full_connect_layer::initialize_weights() {
		weights_.resize(out_size_, in_size_);
		biases_.resize(out_size_);
		d_feature_vector_.resize(out_size_);
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::normal_distribution<double> distribution(0.0, 0.01);
		for (size_t i = 0; i < out_size_; ++i) {
			for (size_t j = 0; j < in_size_; ++j) {
				weights_(i, j) = distribution(generator);
			}
			biases_(i) = distribution(generator);
		}
	}

	le_net5::le_net5() : conv1_(1, 5, 6, 2, 1, activation_function_type::lrelu), pool1_(2, 2), conv2_(6, 5, 16, 0, 1, activation_function_type::lrelu), pool2_(2, 2),
		fc1_(400, 120, activation_function_type::lrelu), fc2_(120, 84, activation_function_type::lrelu), fc3_(84, 10, activation_function_type::lrelu) {
	}
	Eigen::VectorXd le_net5::forward(const std::vector<Eigen::MatrixXd>& input) {
		auto fm1 = conv1_.forward(input);
		auto fm2 = pool1_.forward(fm1);
		auto fm3 = conv2_.forward(fm2);
		auto fm4 = pool2_.forward(fm3);
		size_t total_elements = 0;
		for (const auto& matrix : fm4) {
			total_elements += matrix.rows() * matrix.cols();
		}
		Eigen::VectorXd flattened = fc1_.get_flatten(fm4);
		fc1_.forward(flattened);
		fc2_.forward(fc1_.feature_vector());
		fc3_.forward(fc2_.feature_vector());
		return fc3_.feature_vector();
	}
	double le_net5::train(const std::vector<std::vector<Eigen::MatrixXd>>& dataset, const std::vector<short>& labels, size_t epochs, double learning_rate) {
		auto total_start_time = std::chrono::high_resolution_clock::now();
		double sum_accuracy = 0.0;
		std::cout << "本次数据：\n";
		int k = 0, l = sqrt(labels.size());
		for (int i : labels) {
			std::cout << i << ' ';
			if (++k >= l) {
				std::cout << "\n";
				k = 0;
			}
		}
		if (k > 0) {
			std::cout << "\n";
		}
		for (size_t epoch = 0; epoch < epochs; ++epoch) {
			auto epoch_start_time = std::chrono::high_resolution_clock::now();
			double loss = 0.0;
			size_t correct = 0;
			std::cout << "迭代次数: " << epoch + 1 << "/" << epochs << std::endl;
			size_t total_samples = dataset.size();
			size_t progress_interval = std::max(total_samples / 140, size_t(1));
			size_t running_count = 0;
			auto last_progress_time = std::chrono::high_resolution_clock::now();
			for (size_t i = 0; i < dataset.size(); ++i) {
				Eigen::VectorXd output = forward(dataset[i]);
				int predicted = predict(output);
				double sample_loss = cross_entropy_loss(output, labels[i]);
				if (predicted == labels[i]) {
					correct++;
				}
				loss += sample_loss;
				running_count++;
				backward(labels[i], learning_rate);
				if ((i + 1) % progress_interval == 0 || (i + 1) == total_samples) {
					auto current_time = std::chrono::high_resolution_clock::now();
					auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - epoch_start_time).count() / 1000.0;
					double progress = static_cast<double>(i + 1) / total_samples * 100.0;
					double current_loss = loss / running_count;
					double current_accuracy = static_cast<double>(correct) / running_count * 100.0;
					double estimated_remaining = 0.0;
					if (progress > 0) {
						estimated_remaining = elapsed_time / progress * (100 - progress);
					}
					std::cout << "\r[";
					int bar_width = 30;
					int pos = static_cast<int>(bar_width * progress / 100.0);
					for (int j = 0; j < bar_width; ++j) {
						if (j < pos) std::cout << "=";
						else std::cout << " ";
					}
					std::cout << "] " << std::setw(3) << static_cast<int>(progress) << "%"
						<< " - 损失: " << std::fixed << std::setprecision(4) << current_loss
						<< " - 准确度: " << std::fixed << std::setprecision(1) << current_accuracy << "%"
						<< " (" << correct << "/" << running_count << ")"
						<< " - 已用时间: " << std::fixed << std::setprecision(1) << elapsed_time << "秒"
						<< " - 预计剩余: " << std::fixed << std::setprecision(1) << estimated_remaining << "秒" << std::flush;
					last_progress_time = current_time;
				}
			}
			auto epoch_end_time = std::chrono::high_resolution_clock::now();
			auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time).count() / 1000.0;
			double avg_loss = loss / dataset.size();
			double accuracy = static_cast<double>(correct) / dataset.size() * 100.0;
			auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - total_start_time).count() / 1000.0;
			double avg_epoch_time = total_elapsed / (epoch + 1);
			double estimated_total_time = avg_epoch_time * epochs;
			double remaining_time = estimated_total_time - total_elapsed;
			std::cout << std::endl << "迭代次数 " << epoch + 1 << " 完成，用时 "
				<< std::fixed << std::setprecision(2) << epoch_duration << "秒: "
				<< "损失 = " << std::fixed << std::setprecision(4) << avg_loss
				<< ", 准确度 = " << std::fixed << std::setprecision(2) << accuracy << "%"
				<< " (" << correct << "/" << dataset.size() << ")" << std::endl;
			std::cout << "总用时: " << std::fixed << std::setprecision(2) << total_elapsed << "秒, "
				<< "预计剩余: " << std::fixed << std::setprecision(2) << remaining_time << "秒"
				<< std::endl << std::endl;
			sum_accuracy += accuracy;
		}
		auto total_end_time = std::chrono::high_resolution_clock::now();
		auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time).count() / 1000.0;
		std::cout << "训练完成，总用时 " << std::fixed << std::setprecision(2) << total_duration
			<< " 秒 (" << std::fixed << std::setprecision(2) << total_duration / 60.0 << " 分钟)"
			<< std::endl;
		return sum_accuracy /= epochs;
	}
	int le_net5::predict(const Eigen::VectorXd& output) {
		int max_index = 0;
		double max_value = output[0];
		for (int i = 1; i < output.size(); ++i) {
			if (output[i] > max_value) {
				max_value = output[i];
				max_index = i;
			}
		}
		return max_index;
	}
	void le_net5::save(const std::filesystem::path& path) {
		std::filesystem::create_directories(path);
		conv1_.save(path / "conv1");
		conv2_.save(path / "conv2");
		fc1_.save(path / "fc1.txt");
		fc2_.save(path / "fc2.txt");
		fc3_.save(path / "fc3.txt");
	}
	void le_net5::load(const std::filesystem::path& path) {
		conv1_.load(path / "conv1");
		conv2_.load(path / "conv2");
		fc1_.load(path / "fc1.txt");
		fc2_.load(path / "fc2.txt");
		fc3_.load(path / "fc3.txt");
	}
	void le_net5::backward(int label, double learning_rate) {
		Eigen::VectorXd empty_loss;
		fc3_.backward(empty_loss, learning_rate, true, label);
		auto& fc3_loss = fc3_.loss();
		auto fc2_loss = fc2_.backward(fc3_loss, learning_rate);
		auto fc1_loss = fc1_.backward(fc2_loss, learning_rate);
		auto& pool2_fm = pool2_.feature_map();
		size_t expected_size = 0;
		for (const auto& matrix : pool2_fm) {
			expected_size += matrix.rows() * matrix.cols();
		}
		if (fc1_loss.size() != expected_size) {
			// throw std::runtime_error("维度不匹配！FC1损失大小与预期大小不符");
			Eigen::VectorXd adjusted_loss = Eigen::VectorXd::Zero(expected_size);
			size_t copy_size = std::min(size_t(fc1_loss.size()), expected_size);
			adjusted_loss.head(copy_size) = fc1_loss.head(copy_size);
			fc1_loss = adjusted_loss;
		}
		auto pool2_input_loss = safe_vector_to_feature_map(fc1_loss, pool2_fm);
		if (pool2_input_loss.empty()) {
			// throw std::runtime_error("将向量转换为特征图失败");
		}
		auto pool2_output_loss = pool2_.backward(pool2_input_loss);
		auto conv2_output_loss = conv2_.backward(pool2_output_loss);
		auto pool1_output_loss = pool1_.backward(conv2_output_loss);
		auto conv1_output_loss = conv1_.backward(pool1_output_loss);
		conv1_.weights_update(learning_rate, conv1_output_loss);
		conv2_.weights_update(learning_rate, conv2_output_loss);
	}
	double le_net5::cross_entropy_loss(const Eigen::VectorXd& output, int label) {
		Eigen::VectorXd softmax_output = output.array().exp();
		softmax_output /= softmax_output.sum();
		return -std::log(softmax_output[label] + 1e-8);
	}
	std::vector<Eigen::MatrixXd> le_net5::vector_to_feature_map(const Eigen::VectorXd& vec, size_t channels, size_t rows, size_t cols) {
		std::vector<Eigen::MatrixXd> result;
		size_t index = 0;
		for (size_t i = 0; i < channels; ++i) {
			Eigen::MatrixXd channel(rows, cols);
			for (size_t r = 0; r < rows; ++r) {
				for (size_t c = 0; c < cols; ++c) {
					channel(r, c) = vec[index++];
				}
			}
			result.push_back(channel);
		}
		return result;
	}
	std::vector<Eigen::MatrixXd> le_net5::safe_vector_to_feature_map(const Eigen::VectorXd& vec, const std::vector<Eigen::MatrixXd>& target_shape) {
		std::vector<Eigen::MatrixXd> result;
		size_t total_elements = 0;
		for (const auto& matrix : target_shape) {
			total_elements += matrix.rows() * matrix.cols();
		}
		if (vec.size() != total_elements) {
			// throw std::runtime_error("向量大小与目标形状总元素数不匹配");
			return result;
		}
		size_t index = 0;
		for (const auto& matrix : target_shape) {
			size_t rows = matrix.rows();
			size_t cols = matrix.cols();
			Eigen::MatrixXd channel(rows, cols);
			for (size_t r = 0; r < rows; ++r) {
				for (size_t c = 0; c < cols; ++c) {
					if (index < vec.size()) {
						channel(r, c) = vec[index++];
					}
					else {
						// throw std::runtime_error("转换过程中索引超出范围");
						channel(r, c) = 0.0;
					}
				}
			}
			result.push_back(channel);
		}
		return result;
	}
}