#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "cnn_base.h"
#include <QElapsedTimer>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QProgressBar>
#include <QSettings>
#include <QtConcurrentRun>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void update_color();
    void canvas_undo();
    void canvas_redo();
    void canvas_clean();
    void canvas_clear();
    void recognize_digits();
    void export_digits();
    void import_picture();

    void create_model();
    void load_model();
    void save_model_as();
    void train_model(bool show_detail);
    void stop_train();
    void model_inform(const QString& output);
    void model_train_details(double progress, double loss, size_t correct, size_t total);
    void show_about();
    void show_help();

    void model_train();
    void train_data_browse();
    void train_label_browse();
    void test_data_browse();
    void test_label_browse();
    void load_settings();
    void save_settings();

    void update_output(const QString& message);
    void update_train_button(const QString& text);
    void update_status_message(const QString& message);
    void handle_training_finished();
    void handle_training_errored(const QString& error);
    void update_ui_language(bool changed, const QString& path);

    void set_font_path();
    void set_font_point_size();
    void switch_font_bold();
    void switch_font_italic();

    void convert();
    void merge();

signals:
    void output_message(const QString& message);
    void train_button_text_changed(const QString& text);
    void status_message_changed(const QString& message);
    void training_finished();
    void training_errored(const QString& error);

private:
    void initialize();
    bool eventFilter(QObject* obj, QEvent* event) override;
    void timerEvent(QTimerEvent* event) override;
    void closeEvent(QCloseEvent* event) override;
    void draw_line_to(const QPoint& end_point);
    void output_log(const QString& output);
    void update_font();

private:
    Ui::MainWindow* ui_;
    QSettings* settings_;
    QFuture<void> training_;
    bool cancel_training_ = 0;
    bool is_training_ = 0;
    std::unique_ptr<chr::cnn_base> model_;

    int font_id_ = -1;
    QString font_path_;
    int font_point_size_ = 12;
    bool font_bold_ = 0;
    bool font_italic_ = 0;

    bool need_update_ = 0;
    QLabel* status_message_;
    std::vector<chr::image_process::digit_block> digits_;
    std::vector<size_t> digit_labels_;

    QColor color_;
    QPoint point_;
    bool mouse_down_ = 0;
    QImage canvas_;
    std::list<QImage> undo_buffer_;
    std::list<QImage> redo_buffer_;
};

#endif // MAINWINDOW_H
