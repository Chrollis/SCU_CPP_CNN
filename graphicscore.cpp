#include "graphicscore.h"
#include "le_net5.hpp"
#include "ui_graphicscore.h"
#include "vgg16.hpp"
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QSlider>

GraphicsCore::GraphicsCore(QWidget* parent)
    : QMainWindow(parent)
    , ui_(new Ui::GraphicsCore)
{
    ui_->setupUi(this);
    this->startTimer(33); // 定时器，33ms刷新一次
    settings_ = new QSettings("./config.ini", QSettings::IniFormat);
    load_settings(); // 加载配置文件
    this->setWindowTitle("Chrollis's CNN Handwritten Digit Recognition Application");
    status_message_ = new QLabel(this);
    ui_->statusbar->addWidget(status_message_);
    // 画布事件过滤和属性设置
    ui_->canvas->installEventFilter(this);
    ui_->canvas->setAttribute(Qt::WA_StaticContents);
    ui_->canvas->setMouseTracking(1);
    // 初始化画布
    canvas_ = QImage(ui_->canvas->size(), QImage::Format_RGB32);
    canvas_.fill(Qt::white);
    update_color();
    // 连接信号槽
    connect(this, &GraphicsCore::output_message, this, &GraphicsCore::update_output);
    connect(this, &GraphicsCore::train_button_text_changed, this, &GraphicsCore::update_train_button);
    connect(this, &GraphicsCore::status_message_changed, this, &GraphicsCore::update_status_message);
    connect(this, &GraphicsCore::training_finished, this, &GraphicsCore::handle_training_finished);
    // 连接菜单和按钮动作
    connect(ui_->action_recognize, &QAction::triggered, this, &GraphicsCore::recognize_digits);
    connect(ui_->recognize, &QPushButton::clicked, this, &GraphicsCore::recognize_digits);
    connect(ui_->action_export, &QAction::triggered, this, &GraphicsCore::export_digits);
    connect(ui_->action_import_picture, &QAction::triggered, this, &GraphicsCore::import_picture);
    connect(ui_->picture_browse, &QPushButton::clicked, this, &GraphicsCore::import_picture);
    connect(ui_->action_about, &QAction::triggered, this, &GraphicsCore::show_about);
    connect(ui_->action_help, &QAction::triggered, this, &GraphicsCore::show_help);
    connect(ui_->action_close, &QAction::triggered, this, [&]() { exit(0); });
    connect(ui_->red, &QSlider::valueChanged, this, &GraphicsCore::update_color);
    connect(ui_->green, &QSlider::valueChanged, this, &GraphicsCore::update_color);
    connect(ui_->blue, &QSlider::valueChanged, this, &GraphicsCore::update_color);
    connect(ui_->action_save, &QAction::triggered, this, &GraphicsCore::save_settings);
    connect(ui_->action_clear_output, &QAction::triggered, ui_->output, &QTextEdit::clear);
    // 画布操作连接
    connect(ui_->action_clean, &QAction::triggered, this, &GraphicsCore::canvas_clean);
    connect(ui_->clean, &QPushButton::clicked, this, &GraphicsCore::canvas_clean);
    connect(ui_->action_redo, &QAction::triggered, this, &GraphicsCore::canvas_redo);
    connect(ui_->redo, &QPushButton::clicked, this, &GraphicsCore::canvas_redo);
    connect(ui_->action_undo, &QAction::triggered, this, &GraphicsCore::canvas_undo);
    connect(ui_->undo, &QPushButton::clicked, this, &GraphicsCore::canvas_undo);
    connect(ui_->action_clear, &QAction::triggered, this, &GraphicsCore::canvas_clear);
    // 模型操作连接
    connect(ui_->action_new_model, &QAction::triggered, this, &GraphicsCore::create_model);
    connect(ui_->action_open_model, &QAction::triggered, this, &GraphicsCore::load_model);
    connect(ui_->model_browse, &QPushButton::clicked, this, &GraphicsCore::load_model);
    connect(ui_->action_save_as, &QAction::triggered, this, &GraphicsCore::save_model_as);
    connect(ui_->model_train, &QPushButton::clicked, this, &GraphicsCore::model_train);
    connect(ui_->action_train_detailed, &QAction::triggered, this, [&]() { train_model(1); });
    connect(ui_->action_train_simple, &QAction::triggered, this, [&]() { train_model(0); });
    connect(ui_->action_stop_train, &QAction::triggered, this, &GraphicsCore::stop_train);
    // 数据文件浏览连接
    connect(ui_->train_data_browse, &QPushButton::clicked, this, &GraphicsCore::train_data_browse);
    connect(ui_->train_label_browse, &QPushButton::clicked, this, &GraphicsCore::train_label_browse);
    connect(ui_->test_data_browse, &QPushButton::clicked, this, &GraphicsCore::test_data_browse);
    connect(ui_->test_label_browse, &QPushButton::clicked, this, &::GraphicsCore::test_label_browse);
}

GraphicsCore::~GraphicsCore()
{
    delete settings_;
    delete ui_;
}
// 更新画笔颜色
void GraphicsCore::update_color()
{
    int r = ui_->red->value();
    int g = ui_->green->value();
    int b = ui_->blue->value();
    color_ = QColor(r, g, b);
    ui_->label_brush_color->setText(QString("RGB(%1,%2,%3)").arg(r).arg(g).arg(b));
    QPixmap color_block(20, 20);
    color_block.fill(color_);
    ui_->brush_color->setPixmap(color_block);
    need_update_ = 1;
}
// 画布撤销操作
void GraphicsCore::canvas_undo()
{
    if (undo_buffer_.size() > 0) {
        redo_buffer_.push_front(std::move(canvas_));
        canvas_ = std::move(undo_buffer_.front());
        undo_buffer_.pop_front();
        if (redo_buffer_.size() > 16) {
            redo_buffer_.pop_back();
        }
        need_update_ = 1;
        output_log("Canvas operation undone");
    }
}
// 画布重做操作
void GraphicsCore::canvas_redo()
{
    if (redo_buffer_.size() > 0) {
        undo_buffer_.push_front(std::move(canvas_));
        canvas_ = std::move(redo_buffer_.front());
        redo_buffer_.pop_front();
        if (undo_buffer_.size() > 16) {
            undo_buffer_.pop_back();
        }
        need_update_ = 1;
        output_log("Canvas operation redone");
    }
}
// 清空画布
void GraphicsCore::canvas_clean()
{
    undo_buffer_.push_front(canvas_.copy());
    canvas_.fill(Qt::white);
    redo_buffer_.clear();
    if (undo_buffer_.size() > 16) {
        undo_buffer_.pop_back();
    }
    need_update_ = 1;
    output_log("Canvas cleared");
}
// 清空画布历史记录
void GraphicsCore::canvas_clear()
{
    undo_buffer_.clear();
    redo_buffer_.clear();
    output_log("Canvas history cleared");
}
// 识别数字
void GraphicsCore::recognize_digits()
{
    if (training_.isRunning()) {
        QMessageBox::warning(this, "Warning", "It's training now");
        return;
    }
    cv::Mat src = chr::image_process::qimage_to_cv_mat(canvas_);
    digits_ = chr::image_process::process_image(src); // 图像处理提取数字区域
    std::ostringstream oss;
    oss << "Digits recognized: ";
    if (digits_.empty()) {
        oss << "nothing found";
    } else {
        digit_labels_.clear();
        src = chr::image_process::labelize_image(src, digits_);
        QImage dst = chr::image_process::cv_mat_to_qimage(src);
        QPainter painter(&dst);
        painter.setBrush(Qt::yellow);
        for (const auto& digit : digits_) {
            size_t label = model_->predict(model_->forward({ digit.data })); // 使用模型预测数字
            digit_labels_.push_back(label);
            oss << label << "; ";
            painter.setPen(Qt::NoPen); // 在画布上绘制识别结果
            painter.drawRect(digit.rect.x, digit.rect.y + digit.rect.height - 15, 8, 15);
            painter.setPen(Qt::SolidLine);
            painter.drawText(digit.rect.x, digit.rect.y + digit.rect.height, QString("%1").arg(label));
        }
        undo_buffer_.push_front(canvas_.copy());
        canvas_ = std::move(dst);
        redo_buffer_.clear();
        if (undo_buffer_.size() > 16) {
            undo_buffer_.pop_back();
        }
        need_update_ = 1;
    }
    output_log(oss.str().c_str());
}
// 导出识别到的数字
void GraphicsCore::export_digits()
{
    if (digits_.empty()) {
        QMessageBox::information(this, "Information", "There's no recognized digits");
        return;
    }
    QString path = QFileDialog::getExistingDirectory(this, "Choose export directory", ".");
    if (path.isEmpty())
        return;
    for (size_t i = 0; i < digits_.size(); i++) {
        cv::Mat mat = chr::image_process::eigen_matrix_to_cv_mat(digits_[i].data);
        cv::Mat resized;
        cv::resize(mat, resized, cv::Size(160, 160));
        cv::imshow(std::to_string(digit_labels_[i]), resized);
        bool flag = 0; // 询问用户正确的标签
        size_t real = static_cast<size_t>(QInputDialog::getInt(this, "Correct Label", "The true label you think: ", static_cast<int>(digit_labels_[i]), 0, 9, 1, &flag));
        if (!flag) {
            cv::imwrite(QString("%1/%2-%3.png").arg(path).arg(digit_labels_[i]).arg(digits_[i].hash).toStdString(), mat);
        } else {
            cv::imwrite(QString("%1/%2-%3.png").arg(path).arg(real).arg(digits_[i].hash).toStdString(), mat);
        }
        cv::destroyWindow(std::to_string(digit_labels_[i]));
    }
    output_log("Digits has been exported to " + path);
}
// 导入图片
void GraphicsCore::import_picture()
{
    QString path = QFileDialog::getOpenFileName(this, "Choose the picture", ".", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (path.isEmpty())
        return;
    ui_->picture_path->setText(path);
    QImage image(path);
    if (image.isNull()) {
        QMessageBox::warning(this, "Error", "Failed to import picture");
        return;
    }
    undo_buffer_.push_front(canvas_.copy());
    image = image.scaled(ui_->canvas->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPainter painter(&canvas_);
    painter.drawImage(0, 0, image);
    redo_buffer_.clear();
    if (undo_buffer_.size() > 16) {
        undo_buffer_.pop_back();
    }
    need_update_ = 1;
    output_log("Picture has been imported from" + path);
}
// 创建模型
void GraphicsCore::create_model()
{
    if (training_.isRunning()) {
        QMessageBox::warning(this, "Warning", "Training is in progress, please wait for it to complete");
        return;
    }
    QString path = QFileDialog::getSaveFileName(this, "Create New Model", ".", "CNN Model Files (*.cnn)");
    if (path.isEmpty()) {
        return;
    }
    // 确保文件名以.cnn结尾
    if (!path.endsWith(".cnn", Qt::CaseInsensitive)) {
        path += ".cnn";
    }
    ui_->model_path->setText(path);
    save_settings();
    try {
        // 根据选择的模型类型创建相应模型
        if (ui_->model_type->currentText() == "Le-Net5") {
            model_ = std::make_unique<chr::le_net5>();
        } else if (ui_->model_type->currentText() == "VGG16") {
            model_ = std::make_unique<chr::vgg16>();
        } else {
            throw std::runtime_error("Invalid model type");
        }
        // 保存新模型
        connect(model_.get(), &chr::cnn_base::inform, this, &GraphicsCore::model_inform);
        connect(model_.get(), &chr::cnn_base::train_details, this, &GraphicsCore::model_train_details);
        model_->save_binary(path.toStdString());
    } catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", QString("Failed to create model: ") + e.what());
        model_.reset(); // 重置模型指针
    }
}
// 加载模型
void GraphicsCore::load_model()
{
    QString path = QFileDialog::getOpenFileName(this, "Choose model file", ".", "CNN Model Files (*.cnn)");
    if (path.isEmpty())
        return;
    ui_->model_path->setText(path);
    save_settings();
    try {
        // 根据选择的模型类型创建相应模型
        if (ui_->model_type->currentText() == "Le-Net5") {
            model_ = std::make_unique<chr::le_net5>();
        } else if (ui_->model_type->currentText() == "VGG16") {
            model_ = std::make_unique<chr::vgg16>();
        } else {
            throw std::runtime_error("Invalid model type");
        }
        if (model_) {
            connect(model_.get(), &chr::cnn_base::inform, this, &GraphicsCore::model_inform);
            connect(model_.get(), &chr::cnn_base::train_details, this, &GraphicsCore::model_train_details);
            model_->load_binary(path.toStdString()); // 加载已有模型
        }
    } catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", QString("Failed to load model: ") + e.what());
        model_.reset();
    }
}
// 模型另存为
void GraphicsCore::save_model_as()
{
    if (!model_) {
        QMessageBox::warning(this, "Warning", "No model loaded to save");
        return;
    }
    QString path = QFileDialog::getSaveFileName(this, "Save Model As", ".", "CNN Model Files (*.cnn)");
    if (path.isEmpty()) {
        return;
    }
    // 确保文件名以.cnn结尾
    if (!path.endsWith(".cnn", Qt::CaseInsensitive)) {
        path += ".cnn";
    }
    ui_->model_path->setText(path);
    save_settings();
    try {
        model_->save_binary(path.toStdString());
    } catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", QString("Failed to save model: ") + e.what());
    }
}
// 训练模型
void GraphicsCore::train_model(bool show_detail)
{
    if (!model_) {
        QMessageBox::warning(this, "Warning", "Please choose or create a model");
        return;
    }
    if (training_.isRunning()) {
        QMessageBox::warning(this, "Warning", "It's training now");
        return;
    }
    try {
        bool flag = 0; // 训练模型
        double target = QInputDialog::getDouble(this, "Train parametre", "Target accuracy(%): ", 0, 0, 100, 4, &flag);
        if (!flag)
            return;
        ui_->model_train->setText("Stop Training");
        is_training_ = 1;
        // 在后台线程中执行训练
        training_ = QtConcurrent::run([this, target, show_detail]() {
            emit output_message("Loading train data...");
            auto train_dataset = chr::mnist_data::obtain_data(
                ui_->train_data->text().toStdString(),
                ui_->train_label->text().toStdString());
            emit output_message("Loading test data...");
            auto test_dataset = chr::mnist_data::obtain_data(
                ui_->test_data->text().toStdString(),
                ui_->test_label->text().toStdString());
            std::vector<std::vector<chr::mnist_data>> train_batches;
            std::vector<chr::mnist_data> batch;
            size_t batch_size = ui_->batch->text().toULongLong();
            // 分批处理训练数据
            for (auto& data : train_dataset) {
                batch.push_back(std::move(data));
                if (batch.size() >= batch_size) {
                    train_batches.push_back(std::move(batch));
                    batch.clear();
                }
            }
            if (batch.size() > 0) {
                train_batches.push_back(std::move(batch));
            }
            emit output_message("Train data have been loaded");
            emit output_message("Test data have been loaded");
            emit output_message(QString("Starting model training - target accuracy: %1%").arg(target));
            QMetaObject::invokeMethod(this, "save_settings", Qt::QueuedConnection);
            size_t epochs = ui_->epoch->text().toULongLong();
            double learning_rate = ui_->learning_rate->text().toDouble();
            double accuracy = 0;
            size_t k = 0;
            // 训练循环，直到达到目标精度或取消训练
            do {
                accuracy = model_->train(train_batches[k], epochs, learning_rate, show_detail);
                k = (k + 1) % train_batches.size();
                model_->save_binary(ui_->model_path->text().toStdString());
            } while (accuracy < target && !cancel_training_);
            if (!cancel_training_) {
                size_t correct = 0;
                emit output_message("Starting model evaluation");
                // 在测试集上评估模型
                for (size_t i = 0; i < test_dataset.size() && !cancel_training_; i++) {
                    auto vec = model_->forward({ test_dataset[i].image() });
                    if (model_->predict(vec) == test_dataset[i].label()) {
                        correct++;
                    }
                    if (show_detail) {
                        double progress = static_cast<double>(i + 1) / test_dataset.size() * 100.0;
                        QMetaObject::invokeMethod(this, "model_train_details",
                            Qt::QueuedConnection,
                            Q_ARG(double, progress),
                            Q_ARG(double, std::numeric_limits<double>::quiet_NaN()),
                            Q_ARG(size_t, correct),
                            Q_ARG(size_t, i + 1));
                    }
                }
                if (!cancel_training_) {
                    double final_accuracy = static_cast<double>(correct) / test_dataset.size() * 100.0;
                    emit output_message(QString("Final test accuracy: %1% (%2/%3)")
                            .arg(final_accuracy, 0, 'f', 2)
                            .arg(correct)
                            .arg(test_dataset.size()));
                }
            }
            emit training_finished();
        });
    } catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", QString("Training failed: ") + e.what());
        is_training_ = false;
        cancel_training_ = false;
        ui_->model_train->setText("Start Training");
    }
}
// 停止训练
void GraphicsCore::stop_train()
{
    if (!training_.isRunning()) {
        QMessageBox::information(this, "Information", "No model is training");
    } else {
        if (QMessageBox::question(this, "Question", "Confirm to stop training") == QMessageBox::Yes) {
            cancel_training_ = 1;
            output_log("Training stop requested - waiting for current batch to complete");
        }
    }
}
// 模型信息输出
void GraphicsCore::model_inform(const std::string& output)
{
    output_log(QString::fromStdString(output));
}
// 训练详情更新
void GraphicsCore::model_train_details(double progress, double loss, size_t correct, size_t total)
{
    std::ostringstream oss;
    int pos = static_cast<int>(40 * progress / 100.0);
    for (int j = 0; j < 40; ++j) {
        if (j < pos)
            oss << "=";
        else
            oss << " ";
    }
    double accuracy = static_cast<double>(correct) / total * 100.0;
    status_message_->setText(QString("%1 %2%, loss: %3, accuracy: %4% (%5/%6)")
            .arg("[" + oss.str() + "]")
            .arg(progress, 0, 'f', 2)
            .arg(loss, 0, 'f', 4)
            .arg(accuracy, 0, 'f', 2)
            .arg(correct)
            .arg(total));
}
// 训练按钮点击处理
void GraphicsCore::model_train()
{
    if (!is_training_) {
        train_model(1);
    } else {
        stop_train();
    }
}
// 训练数据浏览
void GraphicsCore::train_data_browse()
{
    ui_->train_data->setText(QFileDialog::getOpenFileName(this, "Choose train image file", ".", "MNIST files (*.idx3-ubyte)"));
    save_settings();
}
// 训练标签浏览
void GraphicsCore::train_label_browse()
{
    ui_->train_label->setText(QFileDialog::getOpenFileName(this, "Choose train label file", ".", "MNIST files (*.idx1-ubyte)"));
    save_settings();
}
// 测试数据浏览
void GraphicsCore::test_data_browse()
{
    ui_->test_data->setText(QFileDialog::getOpenFileName(this, "Choose test image file", ".", "MNIST files (*.idx3-ubyte)"));
    save_settings();
}
// 测试标签浏览
void GraphicsCore::test_label_browse()
{
    ui_->test_label->setText(QFileDialog::getOpenFileName(this, "Choose test label file", ".", "MNIST files (*.idx1-ubyte)"));
    save_settings();
}
// 更新输出信息
void GraphicsCore::update_output(const QString& message)
{
    output_log(message);
}
// 更新训练按钮文本
void GraphicsCore::update_train_button(const QString& text)
{
    ui_->model_train->setText(text);
}
// 更新状态栏消息
void GraphicsCore::update_status_message(const QString& message)
{
    status_message_->setText(message);
}
// 训练完成处理
void GraphicsCore::handle_training_finished()
{
    status_message_->setText("");
    cancel_training_ = false;
    is_training_ = false;
    ui_->model_train->setText("Start Training");
}
// 事件过滤器，处理画布鼠标事件
bool GraphicsCore::eventFilter(QObject* obj, QEvent* event)
{
    if (obj == ui_->canvas) {
        if (event->type() == QEvent::MouseButtonPress) {
            QMouseEvent* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() == Qt::LeftButton) {
                point_ = mouse->pos();
                mouse_down_ = 1;
                undo_buffer_.push_front(canvas_.copy());
                redo_buffer_.clear();
                if (undo_buffer_.size() > 16) {
                    undo_buffer_.pop_back();
                }
            }
            return 1;
        } else if (event->type() == QEvent::MouseMove && mouse_down_) {
            QMouseEvent* mouse = static_cast<QMouseEvent*>(event);
            draw_line_to(mouse->pos());
            return 1;
        } else if (event->type() == QEvent::MouseButtonRelease) {
            QMouseEvent* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() == Qt::LeftButton && mouse_down_) {
                draw_line_to(mouse->pos());
                mouse_down_ = 0;
            }
            return 1;
        } else if (event->type() == QEvent::Paint) {
            QPaintEvent* paint = static_cast<QPaintEvent*>(event);
            QPainter painter(ui_->canvas);
            painter.fillRect(ui_->canvas->rect(), Qt::white);
            painter.drawImage(0, 0, canvas_);
            return 1;
        }
    }
    return QMainWindow::eventFilter(obj, event);
}
// 定时器事件，更新画布
void GraphicsCore::timerEvent(QTimerEvent* event)
{
    if (need_update_) {
        update();
        need_update_ = 0;
    }
}
// 关闭事件处理
void GraphicsCore::closeEvent(QCloseEvent* event)
{
    if (training_.isRunning()) {
        if (QMessageBox::question(this, "Question", "It's training; closing would kill the process") == QMessageBox::Yes) {
            exit(0);
        }
    } else {
        exit(0);
    }
}
// 绘制线条
void GraphicsCore::draw_line_to(const QPoint& end_point)
{
    QPainter painter(&canvas_);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(QPen(color_, 10, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawLine(point_, end_point);
    for (int i = 1; i <= 2; ++i) {
        QPoint intermediate_point = point_ + (end_point - point_) * i / (2 + 1);
        painter.drawPoint(intermediate_point);
    }
    point_ = end_point;
    need_update_ = 1;
}
// 输出日志
void GraphicsCore::output_log(const QString& output)
{
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    ui_->output->append("[" + timestamp + "]: " + output);
}
// 加载设置
void GraphicsCore::load_settings()
{
    ui_->train_data->setText(settings_->value("train_data").toString());
    ui_->train_label->setText(settings_->value("train_label").toString());
    ui_->test_data->setText(settings_->value("test_data").toString());
    ui_->test_label->setText(settings_->value("test_label").toString());
    ui_->model_path->setText(settings_->value("model_path").toString());
    ui_->model_type->setCurrentIndex(settings_->value("model_type").toInt());
    try {
        // 根据选择的模型类型创建相应模型
        if (ui_->model_type->currentText() == "Le-Net5") {
            model_ = std::make_unique<chr::le_net5>();
        } else if (ui_->model_type->currentText() == "VGG16") {
            model_ = std::make_unique<chr::vgg16>();
        } else {
            throw std::runtime_error("Invalid model type");
        }
        if (model_) {
            connect(model_.get(), &chr::cnn_base::inform, this, &GraphicsCore::model_inform);
            connect(model_.get(), &chr::cnn_base::train_details, this, &GraphicsCore::model_train_details);
            model_->load_binary(ui_->model_path->text().toStdString()); // 加载已有模型
        }
    } catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", QString("Failed to load model: ") + e.what());
        model_.reset();
    }
    ui_->batch->setText(settings_->value("batch").toString());
    ui_->epoch->setText(settings_->value("epoch").toString());
    ui_->learning_rate->setText(settings_->value("learning_rate").toString());
}
// 保存设置
void GraphicsCore::save_settings()
{
    settings_->setValue("train_data", ui_->train_data->text());
    settings_->setValue("train_label", ui_->train_label->text());
    settings_->setValue("test_data", ui_->test_data->text());
    settings_->setValue("test_label", ui_->test_label->text());
    settings_->setValue("model_path", ui_->model_path->text());
    settings_->setValue("model_type", ui_->model_type->currentIndex());
    settings_->setValue("batch", ui_->batch->text());
    settings_->setValue("epoch", ui_->epoch->text());
    settings_->setValue("learning_rate", ui_->learning_rate->text());
}

void GraphicsCore::show_about()
{
    QMessageBox::about(this,
        tr("About GraphicsCore"),
        tr("<h3>GraphicsCore - CNN Handwritten Digit Recognition</h3>"
           "<p>Version 1.0.1</p>"
           "<p>This application provides a comprehensive solution for handwritten digit "
           "recognition using Convolutional Neural Networks (CNN).</p>"
           "<p><b>Features:</b></p>"
           "<ul>"
           "<li>Interactive canvas for digit drawing</li>"
           "<li>Support for multiple CNN architectures (LeNet-5, VGG16)</li>"
           "<li>Model training with MNIST dataset</li>"
           "<li>Real-time digit recognition</li>"
           "<li>Batch processing and export capabilities</li>"
           "</ul>"
           "<p>Developed using Qt framework and custom CNN implementation.</p>"
           "<p>© 2025 Chrollis Phrott. All rights reserved.</p>"));
}

// Help对话框
void GraphicsCore::show_help()
{
    QMessageBox::information(this,
        tr("GraphicsCore Help"),
        tr("<h3>Getting Started with GraphicsCore</h3>"

           "<p><b>Basic Workflow:</b></p>"
           "<ol>"
           "<li><b>Setup Model:</b> Choose a model type and load/create a model file</li>"
           "<li><b>Prepare Data:</b> Load MNIST training and test datasets</li>"
           "<li><b>Train Model:</b> Configure parameters and start training</li>"
           "<li><b>Draw & Recognize:</b> Use the canvas to draw digits and recognize them</li>"
           "</ol>"

           "<p><b>Canvas Operations:</b></p>"
           "<ul>"
           "<li><b>Draw:</b> Click and drag on canvas to draw digits</li>"
           "<li><b>Color:</b> Adjust RGB sliders to change brush color</li>"
           "<li><b>Undo/Redo:</b> Use undo/redo buttons or Ctrl+Z/Ctrl+Shift+Z</li>"
           "<li><b>Clear:</b> Clean canvas with Clean button or Ctrl+Shift+C</li>"
           "</ul>"

           "<p><b>Model Training:</b></p>"
           "<ul>"
           "<li><b>Batch Size:</b> Number of samples per training batch</li>"
           "<li><b>Epoch Times:</b> Number of training iterations</li>"
           "<li><b>Learning Rate:</b> Step size for weight updates</li>"
           "<li><b>Target Accuracy:</b> Training stops when this accuracy is reached</li>"
           "</ul>"

           "<p><b>Supported Model Types:</b></p>"
           "<ul>"
           "<li><b>LeNet-5:</b> Classic CNN architecture for digit recognition</li>"
           "<li><b>VGG16:</b> Deeper network with better feature extraction</li>"
           "</ul>"

           "<p><b>File Formats:</b></p>"
           "<ul>"
           "<li><b>Model Files:</b> *.cnn (custom CNN binary format)</li>"
           "<li><b>MNIST Data:</b> *.idx3-ubyte (images), *.idx1-ubyte (labels)</li>"
           "<li><b>Import Images:</b> PNG, JPG, JPEG, BMP</li>"
           "</ul>"

           "<p>Check the output panel for detailed operation logs and training progress.</p>"));
}
