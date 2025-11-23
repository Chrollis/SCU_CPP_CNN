#ifndef CHR_CNN_HPP
#define CHR_CNN_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace chr {

	unsigned swap_endian(unsigned val);
	void read_and_save(const std::filesystem::path& mnist_img_path, const std::filesystem::path& mnist_label_path, size_t num, const std::filesystem::path& output_path);
	void train_data(size_t num);
	void test_data(size_t num);
	std::vector<int> read_labels(const std::string& label_file_path);
	std::vector<std::vector<int>> read_labels(const std::string& label_file_path, size_t batch_size);

	double loss(const Eigen::VectorXd& input);
	Eigen::MatrixXd convolve(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel, size_t stride = 1);
	std::vector<Eigen::MatrixXd> padding(const std::vector<Eigen::MatrixXd>& input, size_t circle_num, double fill_num);
	Eigen::MatrixXd padding(const Eigen::MatrixXd& input, size_t circle_num, double fill_num);

	class data_loader {
	public:
		std::vector<Eigen::MatrixXd> process_single_img(const std::filesystem::path& path);
		std::vector<std::vector<Eigen::MatrixXd>> build_batch(const std::vector<std::filesystem::path>& img_paths);
		std::vector<std::vector<std::vector<Eigen::MatrixXd>>> build_batches(const std::vector<std::filesystem::path>& img_paths, size_t batch_size);
		std::vector<std::vector<Eigen::MatrixXd>> build_batch_from_directory(const std::filesystem::path& img_dir);
		std::vector<std::vector<std::vector<Eigen::MatrixXd>>> build_batches_from_directory(const std::filesystem::path& img_dir, size_t batch_size);
	public:
		cv::Mat preprocess_img(const cv::Mat& src_img);
		Eigen::MatrixXd process_digit(cv::Mat& digit_mat);
		void apply_padding(cv::Mat& img);
		bool is_valid_digit_region(const cv::Rect& rect, const cv::Size& image_size);
		static Eigen::MatrixXd read_image_to_Eigen(const std::filesystem::path& path);
	};

	enum class activation_function_type : uchar {
		sigmoid,
		tanh,
		relu,
		lrelu
	};
	enum class pooling_type : uchar {
		max,
		average
	};
	class activation_function {
	private:
		std::function<double(double)> function;
		std::function<double(double)> derivative;
	public:
		activation_function(activation_function_type type) { change_function(type); }
		activation_function();
		double operator()(double x) { return function(x); }
		double operator[](double x) { return derivative(x); }
		void change_function(activation_function_type type);
		double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }
		double sigmoid_derivative(double x) { return sigmoid(x) * (1 - sigmoid(x)); }
		double tanh(double x) { return std::tanh(x); }
		double tanh_derivative(double x) { return 1 / pow(cosh(x), 2); }
		double relu(double x) { return x > 0 ? x : 0; }
		double relu_derivative(double x) { return x > 0 ? 1 : 0; }
		double lrelu(double x) { return x > 0 ? x : 0.01 * x; }
		double lrelu_derivative(double x) { return x > 0 ? 1 : 0.01; }
	};

	class filter {
	public:
		size_t width = 3;
		size_t channel = 1;
		double bias = 0;
		std::vector<Eigen::MatrixXd> kernels;
		filter() = default;
		filter(size_t channel, size_t width) :width(width), channel(channel) { initialize_gausz(0.01); }
		void initialize_gausz(double stddev);
		void initialize_xavier(size_t input_size);
		void initialize_He(size_t input_size);
		void print_kernels() const;
		void save(const std::filesystem::path& path) const;
		void load(const std::filesystem::path& path);
	};

	class pool_layer {
	private:
		size_t size_;
		size_t stride_;
		size_t padding_;
		pooling_type type_;
		std::vector<Eigen::MatrixXd> input_;
		std::vector<Eigen::MatrixXd> feature_map_;
		std::vector<Eigen::MatrixXd> record_;
	public:
		pool_layer(size_t size, size_t stride, size_t padding = 0, pooling_type type = pooling_type::max) : size_(size), stride_(stride), padding_(padding), type_(type) {}
        const std::vector<Eigen::MatrixXd>& feature_map() { return feature_map_; }
		std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input);
		std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& loss);
	private:
		void max_pooling(const Eigen::MatrixXd& input, Eigen::MatrixXd& output, Eigen::MatrixXd& record) const;
		void average_pooling(const Eigen::MatrixXd& input, Eigen::MatrixXd& output) const;
		Eigen::MatrixXd max_backward(const Eigen::MatrixXd& loss, const Eigen::MatrixXd& record) const;
		Eigen::MatrixXd average_backward(const Eigen::MatrixXd& loss);
		std::vector<Eigen::MatrixXd> remove_padding(const std::vector<Eigen::MatrixXd>& input, size_t padding);
	};

	class full_connect_layer {
    private:
        size_t in_size_;
        size_t out_size_;
        activation_function afunc_;
        Eigen::MatrixXd weights_;
        Eigen::VectorXd biases_;
        Eigen::VectorXd input_;
        Eigen::VectorXd feature_vector_;
        Eigen::VectorXd d_feature_vector_;
        Eigen::VectorXd loss_;
    public:
        full_connect_layer(size_t in_size, size_t out_size, activation_function_type activate_type = activation_function_type::relu) : in_size_(in_size), out_size_(out_size), afunc_(activate_type) { initialize_weights(); }
		const Eigen::VectorXd& loss() const { return loss_; }
		Eigen::VectorXd forward(const Eigen::VectorXd& input);
        Eigen::VectorXd backward(const Eigen::VectorXd& loss, double learning_rate, bool is_output_layer = false, int label = 0);
        void weights_update(double learning_rate);
        Eigen::VectorXd get_flatten(const std::vector<Eigen::MatrixXd>& feature_map);
        const Eigen::VectorXd& feature_vector() const { return feature_vector_; }
        const Eigen::VectorXd& input() const { return input_; }
        void save(const std::filesystem::path& path) const;
        void load(const std::filesystem::path& path);
    private:
        void initialize_weights();
	};

    class convolve_layer {
    private:
        size_t in_channel_;
        size_t kernel_size_;
        size_t out_channel_;
        size_t padding_;
        size_t stride_;
        activation_function afunc_;
        std::vector<filter> filters_;
        std::vector<Eigen::MatrixXd> input_;
        std::vector<Eigen::MatrixXd> feature_map_;
        std::vector<Eigen::MatrixXd> d_feature_map_;
    public:
        convolve_layer(size_t in_channel, size_t kernel_size, size_t out_channel, size_t padding = 0, size_t stride = 1, activation_function_type activate_type = activation_function_type::relu);
        std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input);
        std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& loss, bool is_last_conv = false);
        void weights_update(double learning_rate, const std::vector<Eigen::MatrixXd>& loss);
        const std::vector<Eigen::MatrixXd>& feature_map() const { return feature_map_; }
        const std::vector<Eigen::MatrixXd>& input() const { return input_; }
        void save(const std::filesystem::path& path) const;
        void load(const std::filesystem::path& path);
    private:
        std::vector<Eigen::MatrixXd> apply_activation_derivative(const std::vector<Eigen::MatrixXd>& loss);
		Eigen::MatrixXd compute_weight_gradient(const Eigen::MatrixXd& input, const Eigen::MatrixXd& loss) const { return convolve(input, loss, stride_); }
        std::vector<Eigen::MatrixXd> remove_padding(const std::vector<Eigen::MatrixXd>& input, size_t padding);
    };

    class le_net5 {
    private:
        convolve_layer conv1_;
        pool_layer pool1_;
        convolve_layer conv2_;
        pool_layer pool2_;
        full_connect_layer fc1_;
        full_connect_layer fc2_;
        full_connect_layer fc3_;
    public:
        le_net5();
        Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd>& input);
        double train(const std::vector<std::vector<Eigen::MatrixXd>>& dataset, const std::vector<int>& labels, size_t epochs, double learning_rate);
        int predict(const Eigen::VectorXd& output);
        void save(const std::filesystem::path& path);
        void load(const std::filesystem::path& path);
		void backward(int label, double learning_rate);
    private:
        double cross_entropy_loss(const Eigen::VectorXd& output, int label);
        std::vector<Eigen::MatrixXd> vector_to_feature_map(const Eigen::VectorXd& vec, size_t channels, size_t rows, size_t cols);
        std::vector<Eigen::MatrixXd> safe_vector_to_feature_map(const Eigen::VectorXd& vec, const std::vector<Eigen::MatrixXd>& target_shape);
    };
}

#endif // !CHR_CNN_HPP
