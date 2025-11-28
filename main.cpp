#include "graphicscore.h"
#include <QApplication>
#include <QFont>
#include <QFontDatabase>

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    int font_ID = QFontDatabase::addApplicationFont("./ark-pixel-12px.ttf");
    if (font_ID != -1) {
        QStringList font_fmls = QFontDatabase::applicationFontFamilies(font_ID);
        if (!font_fmls.isEmpty()) {
            QFont main_font(font_fmls.at(0));
            main_font.setPointSize(12);
            QApplication::setFont(main_font);
        }
    }
    Eigen::setNbThreads(omp_get_max_threads());
    GraphicsCore c;
    c.show();
    return a.exec();
}
