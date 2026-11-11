#include <QApplication>
#include <QTimer>
#include <QPixmap>
#include <QString>
#include <QFileInfo>
#include "MainWindow.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    MainWindow window;
    window.resize(1366, 820);
    window.show();

    if (argc > 1) {
        QString arg1 = argv[1];
        if (arg1 == "--test-regions" && argc >= 4) {
            QString cadPath = argv[2];
            QString outPath = argv[3];
            window.loadCADFile(cadPath);
            window.toggleColorRegions();
            QTimer::singleShot(1000, [&]() {
                QPixmap pix = window.grab();
                pix.save(outPath);
                app.quit();
            });
            return app.exec();
        } else if (arg1 == "--screenshot" && argc >= 4) {
            QString cadPath = argv[2];
            QString outPath = argv[3];
            window.loadCADFile(cadPath);
            int tab = (argc >= 5) ? QString(argv[4]).toInt() : 0;
            window.selectTab(tab);
            QTimer::singleShot(1000, [&]() {
                QPixmap pix = window.grab();
                pix.save(outPath);
                app.quit();
            });
            return app.exec();
        } else if (QFileInfo::exists(arg1)) {
            window.loadCADFile(arg1);
        }
    }

    return app.exec();
}
