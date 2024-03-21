import os.path
import pdb

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
from cir_system import Ui_MainWindow
from inference_mpac import *
from inference_pic2cup import *


class MyMainWindow(QMainWindow, Ui_MainWindow, QWidget):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.dataset = 'FashionIQ'

    def choose_refImage(self):
        ref_imgPath, _ = QFileDialog.getOpenFileName(self, "Open Pictures", "", "Image (*.jpg *.png)")
        if ref_imgPath:
            ref_image = QImage(ref_imgPath)
            if ref_image.isNull():
                QMessageBox.information(self, "Error", "Cannot load %s." % ref_imgPath)
            else:
                pixmap = QPixmap.fromImage(ref_image)
                scale = self.adapt2win(self.graphicsView.width(), self.graphicsView.height(),
                                       pixmap.width(), pixmap.height())
                item = QGraphicsPixmapItem(pixmap)
                item.setScale(scale * 0.95)
                scene = QGraphicsScene()
                scene.addItem(item)
                self.graphicsView.setScene(scene)
        self.ref_imgPath = ref_imgPath
        return

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Open Pictures", "", "Model (*.pt)")
        self.model_path = model_path
        self.lineEdit.setText(os.path.basename(model_path))
        return

    def choose_dataset(self):
        if self.comboBox.currentText() == 'FashionIQ':
            self.dataset = 'FashionIQ'
        elif self.comboBox.currentText() == 'CIRR':
            self.dataset = 'CIRR'
        return

    def adapt2win(self, view_w, view_h, img_w, img_h):
        ratio_w = view_w * 1.0 / img_w
        ratio_h = view_h * 1.0 / img_h
        if ratio_w > ratio_h:
            scale = ratio_h
        else:
            scale = ratio_w
        return scale

    def retrieval_and_show(self):
        # mod_text = ['is lighter colored with a panda image', 'is more animal-like and stylish']
        if self.dataset == 'FashionIQ':
            data_dir = 'database/fashionIQ/images'
            mod_text = [self.textEdit.toPlainText(), self.textEdit_2.toPlainText()]
        elif self.dataset == 'CIRR':
            data_dir = 'database/cirr/dev'
            mod_text = self.textEdit.toPlainText()

        model_type = os.path.splitext(os.path.basename(self.model_path))[0]
        if model_type == 'mpac':
            res = inference_mpac(self.dataset, self.model_path, self.ref_imgPath, mod_text)
        elif model_type == 'pic2cup':
            res = inference_pic2cup(self.dataset, self.model_path, self.ref_imgPath, mod_text)

        graphicView = [self.graphicsView_2, self.graphicsView_3, self.graphicsView_4,
                       self.graphicsView_5, self.graphicsView_6, self.graphicsView_7]
        for i in range(len(res)):
            res_path = os.path.join(data_dir, res[i] + '.png')
            if res_path:
                image = QImage(res_path)
                if image.isNull():
                    QMessageBox.information(self, "Error", "Cannot load %s" % res_path)
                else:
                    pixmap = QPixmap.fromImage(image)
                    scale = self.adapt2win(self.graphicsView.width(), self.graphicsView.height(),
                                           pixmap.width(), pixmap.height())
                    item = QGraphicsPixmapItem(pixmap)
                    item.setScale(scale * 0.95)
                    scene = QGraphicsScene()
                    scene.addItem(item)
                    graphicView[i].setScene(scene)
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序对象
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())  # 在主线程中退出
