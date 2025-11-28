#include "vgg16.hpp"
#include <fstream>

namespace chr {
vgg16::vgg16()
    : conv1_(1, 3, 2, 3, 1, activation_function_type::lrelu)
    , // 32*32
    conv2_(2, 3, 2, 1, 1, activation_function_type::lrelu)
    , pool1_(2, 2)
    , // 16*16
    conv3_(2, 3, 4, 1, 1, activation_function_type::lrelu)
    , conv4_(4, 3, 4, 1, 1, activation_function_type::lrelu)
    , pool2_(2, 2)
    , // 8*8
    conv5_(4, 3, 8, 1, 1, activation_function_type::lrelu)
    , conv6_(8, 3, 8, 1, 1, activation_function_type::lrelu)
    , conv7_(8, 3, 8, 1, 1, activation_function_type::lrelu)
    , pool3_(2, 2)
    , // 4*4
    conv8_(8, 3, 16, 1, 1, activation_function_type::lrelu)
    , conv9_(16, 3, 16, 1, 1, activation_function_type::lrelu)
    , conv10_(16, 3, 16, 1, 1, activation_function_type::lrelu)
    , pool4_(2, 2)
    , // 2*2
    conv11_(16, 3, 32, 1, 1, activation_function_type::lrelu)
    , conv12_(32, 3, 32, 1, 1, activation_function_type::lrelu)
    , conv13_(32, 3, 32, 1, 1, activation_function_type::lrelu)
    , pool5_(2, 2)
    , // 1*1
    fc1_(32, 24, activation_function_type::lrelu)
    , fc2_(24, 16, activation_function_type::lrelu)
    , fc3_(16, 10, activation_function_type::lrelu)
{
}
Eigen::VectorXd vgg16::forward(const std::vector<Eigen::MatrixXd>& input)
{
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
std::vector<Eigen::MatrixXd> vgg16::backward(size_t label, double learning_rate)
{
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
void vgg16::save(const std::filesystem::path& path)
{
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
    emit inform("VGG16 model saved to: " + path.string());
}
void vgg16::load(const std::filesystem::path& path)
{
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
    emit inform("VGG16 model loaded from: " + path.string());
}

void vgg16::save_binary(const std::filesystem::__cxx11::path& path)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving: " + path.string());
    }
    // 写入魔数 1128
    uint32_t magic_number = 1128;
    file.write(reinterpret_cast<const char*>(&magic_number), sizeof(magic_number));
    // 写入模型类型名
    std::string model_type = this->model_type();
    uint32_t type_length = model_type.length();
    file.write(reinterpret_cast<const char*>(&type_length), sizeof(type_length));
    file.write(model_type.c_str(), type_length);
    // 保存各层参数
    conv1_.save_binary(file);
    conv2_.save_binary(file);
    conv3_.save_binary(file);
    conv4_.save_binary(file);
    conv5_.save_binary(file);
    conv6_.save_binary(file);
    conv7_.save_binary(file);
    conv8_.save_binary(file);
    conv9_.save_binary(file);
    conv10_.save_binary(file);
    conv11_.save_binary(file);
    conv12_.save_binary(file);
    conv13_.save_binary(file);
    fc1_.save_binary(file);
    fc2_.save_binary(file);
    fc3_.save_binary(file);
    file.close();
    emit inform("VGG16 model saved to: " + path.string());
}

void vgg16::load_binary(const std::filesystem::__cxx11::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading: " + path.string());
    }
    // 读取并验证魔数
    uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    if (magic_number != 1128) {
        throw std::runtime_error("Invalid file format: magic number mismatch");
    }
    // 读取并验证模型类型
    uint32_t type_length;
    file.read(reinterpret_cast<char*>(&type_length), sizeof(type_length));
    std::string model_type(type_length, ' ');
    file.read(&model_type[0], type_length);
    if (model_type != this->model_type()) {
        throw std::runtime_error("Model type mismatch: expected " + this->model_type() + ", got " + model_type);
    }
    // 加载各层参数
    conv1_.load_binary(file);
    conv2_.load_binary(file);
    conv3_.load_binary(file);
    conv4_.load_binary(file);
    conv5_.load_binary(file);
    conv6_.load_binary(file);
    conv7_.load_binary(file);
    conv8_.load_binary(file);
    conv9_.load_binary(file);
    conv10_.load_binary(file);
    conv11_.load_binary(file);
    conv12_.load_binary(file);
    conv13_.load_binary(file);
    fc1_.load_binary(file);
    fc2_.load_binary(file);
    fc3_.load_binary(file);
    file.close();
    emit inform("VGG16 model loaded from: " + path.string());
}
}
