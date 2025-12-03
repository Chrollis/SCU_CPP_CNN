#include "mainwindow.h"
#include <QApplication>
int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    Eigen::setNbThreads(omp_get_max_threads());
    MainWindow w;
    w.show();
    return a.exec();
}
