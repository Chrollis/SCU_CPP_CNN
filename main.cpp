#include "le_net5.hpp"
#include "vgg16.hpp"
#include "ui_form.hpp"

using namespace chr;

void train(std::filesystem::path data_dir, std::filesystem::path model_dir, size_t batch_size, size_t epochs, double target_accuracy = 100, double learning_rate = 0.001, bool show_detail = 0) {
    Eigen::setNbThreads(omp_get_max_threads()); // 设置Eigen使用多线程
    std::cout << "Read the train data..." << std::endl;
    auto train = mnist_data::obtain_data(
        data_dir / "train-images.idx3-ubyte",
        data_dir / "train-labels.idx1-ubyte"
    );
    le_net5 model; // 创建LeNet-5模型
    std::cout << "Load the LeNet-5 model..." << std::endl;
    model.load(model_dir); // 加载已有模型
    std::cout << "Start training..." << std::endl;
    std::cout << "Samples: " << train.size() << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    // 将训练数据分批
    std::vector<std::vector<mnist_data>> train_batches;
    std::vector<mnist_data> batch;
    for (auto& data : train) {
        batch.push_back(std::move(data));
        if (batch.size() >= batch_size) {
            train_batches.push_back(std::move(batch));
            batch.clear();
        }
    }
    if (batch.size() > 0) {
        train_batches.push_back(std::move(batch));
        batch.clear();
    }
    train.clear();
    double accuracy = 0.0;
    do {
        static size_t k = 0;
        // 训练当前批次
        accuracy = model.train(train_batches[k], epochs, learning_rate, show_detail);
        k = (++k) % train_batches.size(); // 循环使用批次
        std::cout << "Save the model..." << std::endl;
        model.save(model_dir); // 保存模型
    } while (accuracy <= target_accuracy); // 达到目标准确率后停止
    // 在测试集上评估模型
    std::cout << "Read the test data..." << std::endl;
    auto test = mnist_data::obtain_data(
        data_dir / "t10k-images.idx3-ubyte",
        data_dir / "t10k-labels.idx1-ubyte"
    );
    std::cout << "Evaluate on the test set..." << std::endl;
    size_t correct = 0;
    for (size_t i = 0; i < test.size(); ++i) {
        auto output = model.forward({ test[i].image() });
        size_t predicted = model.predict(output);
        if (predicted == test[i].label()) {
            correct++;
        }
        accuracy = static_cast<double>(correct) / test.size();
        std::cout << "\rTest accuracy: " << std::to_string(accuracy * 100) << "% ("
            << correct << "/" << test.size() << ")\t";
    }
    std::cout << "\rTest accuracy: " << std::to_string(accuracy * 100) << "% ("
        << correct << "/" << test.size() << ")" << std::endl;
    std::cout << "Acomplished!" << std::endl;
}

IMAGE detect(const IMAGE& src, std::vector<image_process::digit_block>& digits, le_net5& model) {
    cv::Mat cv_img = image_process::easyx_to_image(src);
    digits = image_process::process_image(cv_img);
    cv_img = image_process::labelize_image(cv_img, digits);
    IMAGE dst = image_process::image_to_easyx(cv_img);
    SetWorkingImage(&dst);
    for (const auto& digit : digits) {
        auto vec = model.forward({ digit.data });
        size_t label = model.predict(vec);
        settextcolor(0);
        setbkcolor(0xC0C0C0);
        setbkmode(OPAQUE);
        settextstyle(16, 0, L"Times New Roman");
        outtextxy(digit.rect.x, digit.rect.y, std::to_wstring(label).c_str());
    }
    SetWorkingImage();
    return dst;
}
void import_image(canvas& paper) {
    int dw = paper.width() - 2, dh = paper.height() - 2;
    IMAGE src, dst(dw, dh);
    loadimage(&src, L"target.jpg");
    int sw = src.getwidth(), sh = src.getheight();
    auto dptr = GetImageBuffer(&dst);
    auto sptr = GetImageBuffer(&src);
    for (int i = 0; i < dw; i++) {
        for (int j = 0; j < dh; j++) {
            double x = (double)i * (sw - 1) / (dw - 1);
            double y = (double)j * (sh - 1) / (dh - 1);
            int x1 = (int)x;
            int y1 = (int)y;
            int x2 = min(x1 + 1, sw - 1);
            int y2 = min(y1 + 1, sh - 1);
            double wx = x - x1;
            double wy = y - y1;
            COLORREF c11 = sptr[y1 * sw + x1];
            COLORREF c12 = sptr[y2 * sw + x1];
            COLORREF c21 = sptr[y1 * sw + x2];
            COLORREF c22 = sptr[y2 * sw + x2];
            // 双线性插值
            int r = (int)((1 - wx) * (1 - wy) * GetRValue(c11) +
                (1 - wx) * wy * GetRValue(c12) +
                wx * (1 - wy) * GetRValue(c21) +
                wx * wy * GetRValue(c22));
            int g = (int)((1 - wx) * (1 - wy) * GetGValue(c11) +
                (1 - wx) * wy * GetGValue(c12) +
                wx * (1 - wy) * GetGValue(c21) +
                wx * wy * GetGValue(c22));
            int b = (int)((1 - wx) * (1 - wy) * GetBValue(c11) +
                (1 - wx) * wy * GetBValue(c12) +
                wx * (1 - wy) * GetBValue(c21) +
                wx * wy * GetBValue(c22));
            dptr[j * dw + i] = RGB(r, g, b);
        }
    }
    paper.set_image(dst);
}

void graphic_interface(std::filesystem::path model_dir) {
    initgraph(640, 512);
    setbkcolor(0xC0C0C0);
    std::vector<ui_base*> parts;
    std::vector<image_process::digit_block> digits;
    bool running = 1;
    le_net5 model;
    ExMessage msg = { 0 };
    model.load(model_dir);

    canvas paper({ 0,0 }, 512, 512);
    paper.set_brush(0xFF0000);
    parts.push_back(&paper);

    button exit_btn({ 576,0 }, 64, 64);
    exit_btn.set_text(L"Exit");
    exit_btn.set_function([&running]() { running = 0; });
    parts.push_back(&exit_btn);

    button undo_btn({ 512,0 }, 64, 64);
    undo_btn.set_text(L"Undo");
    undo_btn.set_function([&paper]() { paper.undo(); });
    parts.push_back(&undo_btn);

    button rubber_btn({ 512,64 }, 64, 64);
    rubber_btn.set_text(L"Rubber");
    rubber_btn.set_function([&paper]() { paper.as_rubber(); });
    parts.push_back(&rubber_btn);

    button brush_btn({ 512,128 }, 64, 64);
    brush_btn.set_text(L"Brush");
    brush_btn.set_function([&paper]() { paper.as_brush(); });
    parts.push_back(&brush_btn);

    button detect_btn({ 576,64 }, 64, 64);
    detect_btn.set_text(L"Detect");
    detect_btn.set_function([&paper, &digits, &model]() { paper.set_image(detect(paper.image(), digits, model)); });
    parts.push_back(&detect_btn);

    button redo_btn({ 576,128 }, 64, 64);
    redo_btn.set_text(L"Redo");
    redo_btn.set_function([&paper]() { paper.redo(); });
    parts.push_back(&redo_btn);

    button clean_btn({ 576,192 }, 64, 64);
    clean_btn.set_text(L"Clean");
    clean_btn.set_function([&paper]() { paper.clean(); });
    parts.push_back(&clean_btn);

    button import_btn({ 512,192 }, 64, 64);
    import_btn.set_text(L"Import");
    import_btn.set_function([&paper]() { import_image(paper); });
    parts.push_back(&import_btn);

    while (running) {
        cleardevice();
        BeginBatchDraw();
        for (auto part : parts) {
            part->draw();
        }
        EndBatchDraw();
        while (peekmessage(&msg)) {
            for (auto part : parts) {
                part->input(msg);
            }
        }
        Sleep(10);
    }

    closegraph();
}

int main() {
    // train("MNIST_data", "trained_lenet5_model", 60000, 1, 100.0, 1e-5, 1);
    graphic_interface("trained_lenet5_model");
    return 0;
}
