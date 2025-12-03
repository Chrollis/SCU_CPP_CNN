#include "mnist_data.hpp"
#include "language_manager.h"
#include <fstream>

namespace chr {
mnist_data::mnist_data(const Eigen::MatrixXd& image, size_t label)
    : image_(image)
    , label_(label)
{
}
mnist_data::mnist_data(Eigen::MatrixXd&& image, size_t label)
    : image_(image)
    , label_(label)
{
}
bool mnist_data::is_legal() const
{
    if (image_.rows() != 28 || image_.cols() != 28)
        return 0; // 检查是否为28x28尺寸
    return 1;
}
// 字节序转换函数(大端转小端)
unsigned mnist_data::swap_endian(unsigned val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}
// 验证MNIST文件格式并返回数据项数量
unsigned mnist_data::check_mnist_file(std::ifstream& mnist_images, std::ifstream& mnist_labels)
{
    if (!mnist_images.is_open()) {
        throw std::runtime_error(chr::tr("error.mnist.image_open_failed").toStdString());
    }
    if (!mnist_labels.is_open()) {
        throw std::runtime_error(chr::tr("error.mnist.label_open_failed").toStdString());
    }
    unsigned magic = 0;
    unsigned items = 0, labels = 0;
    // 检查图像文件魔数
    mnist_images.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051) { // MNIST图像文件魔数应为2051
        throw std::runtime_error(chr::tr("error.file.invalid_magic_number").toStdString());
    }
    // 检查标签文件魔数
    mnist_labels.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049) { // MNIST标签文件魔数应为2049
        throw std::runtime_error(chr::tr("error.file.invalid_magic_number").toStdString());
    }
    // 读取数据项数量
    mnist_images.read(reinterpret_cast<char*>(&items), 4);
    items = swap_endian(items);
    mnist_labels.read(reinterpret_cast<char*>(&labels), 4);
    labels = swap_endian(labels);
    if (items != labels) { // 图像和标签数量必须匹配
        throw std::runtime_error(chr::tr("error.mnist.count_mismatch").toStdString());
    }
    return items;
}
// 从MNIST文件读取数据
std::vector<mnist_data> mnist_data::obtain_data(const std::filesystem::path& mnist_image_path, const std::filesystem::path& mnist_label_path, size_t offset, size_t size)
{
    std::ifstream mnist_images(mnist_image_path, std::ios::binary);
    std::ifstream mnist_labels(mnist_label_path, std::ios::binary);
    size_t items = check_mnist_file(mnist_images, mnist_labels); // 验证文件格式
    if (offset >= items)
        return {}; // 偏移量不能超过数据集大小
    std::vector<mnist_data> batch;
    unsigned rows = 0, cols = 0;
    // 读取图像尺寸
    mnist_images.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    mnist_images.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);
    // 跳转到指定偏移量
    mnist_images.seekg(offset * rows * cols, std::ios::cur);
    mnist_labels.seekg(offset, std::ios::cur);
    // 读取指定数量的数据
    for (size_t i = 0; i < size && i + offset < items; i++) {
        Eigen::MatrixXd image(rows, cols);
        std::vector<unsigned char> pixels(rows * cols);
        size_t label = 0;
        // 读取图像像素和标签
        mnist_images.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(rows) * cols);
        mnist_labels.read(reinterpret_cast<char*>(&label), 1);
        // 将像素数据转换为矩阵(二值化)
        for (unsigned m = 0; m < rows; m++) {
            for (unsigned n = 0; n < cols; n++) {
                image(m, n) = pixels[static_cast<size_t>(m) * cols + n] ? 1.0 : 0.0;
            }
        }
        mnist_data piece(std::move(image), label);
        batch.push_back(std::move(piece));
    }
    return batch;
}

void mnist_data::write_data(const std::filesystem::path& image_path, const std::filesystem::path& label_path, const std::vector<mnist_data>& datas)
{
    std::ofstream image_file(image_path, std::ios::binary);
    if (!image_file.is_open()) {
        throw std::runtime_error(chr::tr("error.mnist.image_create_failed").toStdString());
    }
    std::ofstream label_file(label_path, std::ios::binary);
    if (!label_file.is_open()) {
        throw std::runtime_error(chr::tr("error.mnist.label_create_failed").toStdString());
    }
    unsigned int magic_number = swap_endian(2051); // MNIST图像魔数
    unsigned int num = swap_endian(static_cast<unsigned int>(datas.size()));
    unsigned int num_rows = swap_endian(28);
    unsigned int num_cols = swap_endian(28);
    image_file.write(reinterpret_cast<const char*>(&magic_number), 4);
    image_file.write(reinterpret_cast<const char*>(&num), 4);
    image_file.write(reinterpret_cast<const char*>(&num_rows), 4);
    image_file.write(reinterpret_cast<const char*>(&num_cols), 4);
    magic_number = swap_endian(2049); // MNIST标签魔数
    label_file.write(reinterpret_cast<const char*>(&magic_number), 4);
    label_file.write(reinterpret_cast<const char*>(&num), 4);
    for (const auto& data : datas) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                unsigned char pixel = static_cast<unsigned char>(data.image_(i, j) ? 255 : 0);
                image_file.write(reinterpret_cast<const char*>(&pixel), 1);
            }
        }
        unsigned char label_byte = static_cast<unsigned char>(data.label_);
        label_file.write(reinterpret_cast<const char*>(&label_byte), 1);
    }
    image_file.close();
    label_file.close();
}
}
