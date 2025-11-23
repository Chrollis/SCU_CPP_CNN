#ifndef UI_FORM_HPP
#define UI_FORM_HPP

#include "chr_cnn.hpp"

namespace chr {

    // 基本点结构
    struct point {
        int x = -1;
        int y = -1;

        point() = default;
        point(int x, int y) : x(x), y(y) {}

        bool operator==(const point& p) const {
            return x == p.x && y == p.y;
        }

        point& operator=(const point& p) {
            x = p.x;
            y = p.y;
            return *this;
        }
    };

    // UI 控件基类
    class ui_tool {
    protected:
        int left_;
        int top_;
        int width_;
        int height_;

    public:
        ui_tool() = default;
        ui_tool(int left, int top, int width, int height)
            : left_(left), top_(top), width_(width), height_(height) {
        }

        virtual ~ui_tool() = default;

        virtual void render() = 0;
        virtual bool handle_event(const point& mouse_pos, int event_type) = 0;

        bool contains(const point& p) const {
            return p.x >= left_ && p.x <= left_ + width_ &&
                p.y >= top_ && p.y <= top_ + height_;
        }

        virtual void set_position(int left, int top) {
            left_ = left;
            top_ = top;
        }

        virtual void set_size(int width, int height) {
            width_ = width;
            height_ = height;
        }
    };

    // 绘图区域控件
    class drawing_tablet : public ui_tool {
    private:
        bool is_drawing_ = false;
        point start_pos_;
        cv::Mat canvas_;
        int brush_size_ = 25;
        cv::Scalar brush_color_ = cv::Scalar(0, 0, 0); // 黑色

    public:
        drawing_tablet(int left, int top, int width, int height)
            : ui_tool(left, top, width, height) {
            initialize_canvas();
        }

        void initialize_canvas() {
            canvas_ = cv::Mat::zeros(height_, width_, CV_8UC3);
            canvas_.setTo(cv::Scalar(255, 255, 255)); // 白色背景
        }

        void render() override {
            // 在实际实现中，这里会使用图形库显示 canvas_
            // 例如使用 OpenCV 的 imshow 或其它图形库
            cv::imshow("Drawing Canvas", canvas_);
        }

        bool handle_event(const point& mouse_pos, int event_type) override {
            if (!contains(mouse_pos)) return false;

            switch (event_type) {
            case 1: // 鼠标按下
                if (!is_drawing_) {
                    is_drawing_ = true;
                    start_pos_ = mouse_pos;
                    draw_point(mouse_pos);
                    return true;
                }
                break;

            case 2: // 鼠标释放
                if (is_drawing_) {
                    is_drawing_ = false;
                    return true;
                }
                break;

            case 3: // 鼠标移动
                if (is_drawing_) {
                    draw_line(start_pos_, mouse_pos);
                    start_pos_ = mouse_pos;
                    return true;
                }
                break;
            }
            return false;
        }

        void clear() {
            initialize_canvas();
        }

        bool save_image(const std::filesystem::path& path) {
            return cv::imwrite(path.string(), canvas_);
        }

        cv::Mat get_canvas() const { return canvas_; }

        void set_brush_size(int size) { brush_size_ = size; }
        void set_brush_color(const cv::Scalar& color) { brush_color_ = color; }

    private:
        void draw_point(const point& p) const {
            cv::circle(canvas_, cv::Point(p.x - left_, p.y - top_),
                brush_size_ / 2, brush_color_, -1);
        }

        void draw_line(const point& p1, const point& p2) const {
            cv::line(canvas_,
                cv::Point(p1.x - left_, p1.y - top_),
                cv::Point(p2.x - left_, p2.y - top_),
                brush_color_, brush_size_, cv::LINE_AA);
        }
    };

    // 按钮控件
    class ui_button : public ui_tool {
    private:
        std::string text_;
        cv::Scalar text_color_;
        cv::Scalar bg_color_;
        cv::Scalar hover_color_;
        bool is_hovered_ = false;
        std::function<void()> click_callback_;

    public:
        ui_button(int left, int top, int width, int height,
            const std::string& text,
            const cv::Scalar& text_color = cv::Scalar(0, 0, 0),
            const cv::Scalar& bg_color = cv::Scalar(239, 239, 239))
            : ui_tool(left, top, width, height), text_(text),
            text_color_(text_color), bg_color_(bg_color),
            hover_color_(bg_color * 0.8) {
        } // 悬停时变暗

        void render() override {
            // 在实际实现中，这里会绘制按钮
            cv::Mat button_mat(height_, width_, CV_8UC3);
            button_mat.setTo(is_hovered_ ? hover_color_ : bg_color_);

            // 绘制边框
            cv::rectangle(button_mat, cv::Point(0, 0),
                cv::Point(width_ - 1, height_ - 1),
                cv::Scalar(0, 0, 0), 1);

            // 绘制文字（简化版本，实际需要更复杂的文字渲染）
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(text_, cv::FONT_HERSHEY_SIMPLEX,
                0.5, 1, &baseline);
            cv::Point text_org((width_ - text_size.width) / 2,
                (height_ + text_size.height) / 2);

            cv::putText(button_mat, text_, text_org,
                cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color_, 1);

            // 在实际应用中，这里需要将按钮显示到主窗口的相应位置
        }

        bool handle_event(const point& mouse_pos, int event_type) override {
            bool is_inside = contains(mouse_pos);

            switch (event_type) {
            case 3: // 鼠标移动
                is_hovered_ = is_inside;
                return is_inside;

            case 2: // 鼠标释放
                if (is_inside && click_callback_) {
                    click_callback_();
                    return true;
                }
                break;
            }
            return false;
        }

        void set_click_callback(std::function<void()> callback) {
            click_callback_ = std::move(callback);
        }

        void set_text(const std::string& text) { text_ = text; }
    };

    // 标签控件
    class ui_label : public ui_tool {
    private:
        std::string text_;
        cv::Scalar text_color_;
        bool centered_ = false;

    public:
        ui_label(int left, int top, int width, int height,
            const std::string& text,
            const cv::Scalar& text_color = cv::Scalar(0, 0, 0),
            bool centered = false)
            : ui_tool(left, top, width, height), text_(text),
            text_color_(text_color), centered_(centered) {
        }

        void render() override {
            // 在实际实现中，这里会绘制标签
            cv::Mat label_mat(height_, width_, CV_8UC3);
            label_mat.setTo(cv::Scalar(239, 239, 239)); // 浅灰色背景

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(text_, cv::FONT_HERSHEY_SIMPLEX,
                0.5, 1, &baseline);

            cv::Point text_org;
            if (centered_) {
                text_org = cv::Point((width_ - text_size.width) / 2,
                    (height_ + text_size.height) / 2);
            }
            else {
                text_org = cv::Point(5, (height_ + text_size.height) / 2);
            }

            cv::putText(label_mat, text_, text_org,
                cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color_, 1);
        }

        bool handle_event(const point& mouse_pos, int event_type) override {
            return false; // 标签不处理事件
        }

        void set_text(const std::string& text) { text_ = text; }
    };

    // 数字识别 UI 界面
    class digit_recognizer_ui {
    private:
        std::unique_ptr<le_net5> model_;
        std::unique_ptr<data_loader> data_loader_;
        std::vector<std::unique_ptr<ui_tool>> widgets_;

        // 主要控件
        drawing_tablet* drawing_area_;
        ui_button* clear_button_;
        ui_button* recognize_button_;
        ui_label* result_label_;

        const int window_width_ = 800;
        const int window_height_ = 600;
        const int toolbar_width_ = 100;

    public:
        digit_recognizer_ui(const std::filesystem::path& model_path) {
            initialize_model(model_path);
            initialize_ui();
        }

        ~digit_recognizer_ui() = default;

        void run() {
            std::cout << "Digit Recognizer UI Started" << std::endl;
            std::cout << "Usage:" << std::endl;
            std::cout << "- Draw digits in the drawing area" << std::endl;
            std::cout << "- Click 'Clear' to clear the drawing" << std::endl;
            std::cout << "- Click 'Recognize' to analyze the digits" << std::endl;

            // 在实际实现中，这里会运行主事件循环
            // 这里使用控制台模拟交互
            simulate_interaction();
        }

        void process_drawing() {
            // 保存绘图
            std::filesystem::path temp_path = "temp_drawing.jpg";
            if (drawing_area_->save_image(temp_path)) {
                // 使用 data_loader 处理图像
                try {
                    auto digits = data_loader_->process_single_img(temp_path, "output/");
                    std::string result_text = "Recognized: ";

                    for (size_t i = 0; i < digits.size(); ++i) {
                        // 将 Eigen::MatrixXd 转换为 le_net5 需要的输入格式
                        std::vector<Eigen::MatrixXd> input = { digits[i] };
                        auto prediction = model_->forward(input);
                        int recognized_digit = model_->predict(prediction);

                        result_text += std::to_string(recognized_digit);
                        if (i < digits.size() - 1) {
                            result_text += ", ";
                        }
                    }

                    if (digits.empty()) {
                        result_text = "No digits found";
                    }

                    result_label_->set_text(result_text);
                    std::cout << result_text << std::endl;

                }
                catch (const std::exception& e) {
                    std::string error_msg = "Error: " + std::string(e.what());
                    result_label_->set_text(error_msg);
                    std::cerr << error_msg << std::endl;
                }

                // 清理临时文件
                std::filesystem::remove(temp_path);
            }
        }

    private:
        void initialize_model(const std::filesystem::path& model_path) {
            model_ = std::make_unique<le_net5>();
            data_loader_ = std::make_unique<data_loader>();

            if (std::filesystem::exists(model_path)) {
                model_->load(model_path);
                std::cout << "Model loaded from: " << model_path << std::endl;
            }
            else {
                std::cout << "Model not found at: " << model_path << std::endl;
                std::cout << "Using untrained model" << std::endl;
            }
        }

        void initialize_ui() {
            // 创建绘图区域
            auto drawing = std::make_unique<drawing_tablet>(0, 0,
                window_width_ - toolbar_width_,
                window_height_);
            drawing_area_ = drawing.get();
            widgets_.push_back(std::move(drawing));

            // 创建清除按钮
            auto clear_btn = std::make_unique<ui_button>(
                window_width_ - toolbar_width_ + 10, 20, 80, 40, "Clear");
            clear_btn->set_click_callback([this]() {
                drawing_area_->clear();
                result_label_->set_text("Drawing cleared");
                std::cout << "Drawing cleared" << std::endl;
                });
            clear_button_ = clear_btn.get();
            widgets_.push_back(std::move(clear_btn));

            // 创建识别按钮
            auto recognize_btn = std::make_unique<ui_button>(
                window_width_ - toolbar_width_ + 10, 90, 80, 40, "Recognize");
            recognize_btn->set_click_callback([this]() {
                result_label_->set_text("Processing...");
                process_drawing();
                });
            recognize_button_ = recognize_btn.get();
            widgets_.push_back(std::move(recognize_btn));

            // 创建结果标签
            auto result_lbl = std::make_unique<ui_label>(
                window_width_ - toolbar_width_ + 10, 150, 80, 200,
                "Draw digits and click Recognize", cv::Scalar(0, 0, 0), true);
            result_label_ = result_lbl.get();
            widgets_.push_back(std::move(result_lbl));
        }

        void simulate_interaction() {
            // 在实际图形界面中，这里会是事件循环
            // 这里用控制台输入模拟交互

            std::string command;
            while (true) {
                std::cout << "\nCommands: [d]raw, [c]lear, [r]ecognize, [q]uit: ";
                std::cin >> command;

                if (command == "q" || command == "quit") {
                    break;
                }
                else if (command == "c" || command == "clear") {
                    drawing_area_->clear();
                    result_label_->set_text("Drawing cleared");
                    std::cout << "Drawing cleared" << std::endl;
                }
                else if (command == "r" || command == "recognize") {
                    result_label_->set_text("Processing...");
                    process_drawing();
                }
                else if (command == "d" || command == "draw") {
                    std::cout << "Drawing simulation - image saved to temp_drawing.jpg" << std::endl;
                    // 在实际应用中，这里会启动绘图模式
                }
                else {
                    std::cout << "Unknown command" << std::endl;
                }
            }
        }
    };

    // 简化的 UI 管理器
    class ui_manager {
    public:
        static void create_digit_recognizer(const std::filesystem::path& model_path = "model/") {
            digit_recognizer_ui ui(model_path);
            ui.run();
        }

        static void show_image(const std::filesystem::path& image_path) {
            cv::Mat image = cv::imread(image_path.string());
            if (image.empty()) {
                std::cerr << "Cannot load image: " << image_path << std::endl;
                return;
            }

            cv::imshow("Image Viewer", image);
            std::cout << "Press any key to close the image..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    };
}

#endif // !UI_FORM_HPP
