import os.path
import requests
import random
from hashlib import md5
import re
import pdb

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
from cir_system import Ui_MainWindow
from inference_mpac import *
from inference_pic2cup import *


def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

class MyMainWindow(QMainWindow, Ui_MainWindow, QWidget):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)

        # init path
        self.data_dir = 'database/fashionIQ/images'
        self.dataset = 'FashionIQ'
        self.model_path = 'models/fiq_mpac.pt'
        self.model = 'fiq_mpac.pt'

        # init retrieval mode
        self.retrieval_mode = 'it2i' # t2i, i2i, it2i, ii2i

        # input
        self.ref_imgPath = None
        self.mod_text = None
        self.ref_imgPath_list = []

        # init visualization module
        self.ref_scene = QGraphicsScene()
        self.scene = []
        self.graphicView = []
        v = 'self.graphicsView'
        for i in range(6):
            self.scene.append(QGraphicsScene())
            self.graphicView.append(eval(v + '_' +str(i)))

        # init other details
        self.res = None
        self.style = None

    def show_refImage(self, ref_imgPath):
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
                # scene = QGraphicsScene()
                self.ref_scene.clear()
                self.ref_scene.addItem(item)
                self.graphicsView.setScene(self.ref_scene)
        self.ref_imgPath = ref_imgPath
        return

    def choose_refImage(self):
        ref_imgPath, _ = QFileDialog.getOpenFileName(self, "Open Pictures", "", "Image (*.jpg *.png)")
        self.show_refImage(ref_imgPath)
        return

    def reset_refImg(self):
        button_name = self.sender().objectName()
        n = re.findall(r'\d+', button_name)
        n = int(n[0])
        self.ref_scene.clear()
        img_path = os.path.join(self.data_dir, self.res[n] + '.png')
        self.show_refImage(img_path)
        return

    def add_refImgs(self):
        button_name = self.sender().objectName()
        n = re.findall(r'\d+', button_name)
        n = int(n[0])
        img_path = os.path.join(self.data_dir, self.res[n] + '.png')

        if self.sender().isChecked():
            self.ref_imgPath_list.append(img_path)
        else:
            self.ref_imgPath_list.remove(img_path)
        print(self.ref_imgPath_list)
        return

    def load_model(self):
        # model_path, _ = QFileDialog.getOpenFileName(self, "Open Pictures", "", "Model (*.pt)")
        model_dir = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "")
        self.model_path = os.path.join(model_dir, self.model)
        # self.lineEdit.setText(os.path.basename(model_path))
        return

    def load_data_dir(self):
        data_dir = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "")
        if self.dataset == 'FashionIQ':
            self.data_dir = os.path.join(data_dir, 'fashionIQ/images')
        elif self.dataset == 'CIRR':
            self.data_dir = os.path.join(data_dir, 'cirr/dev')
        return

    def choose_dataset(self):
        if self.comboBox.currentText() == 'FashionIQ':
            self.dataset = 'FashionIQ'
        elif self.comboBox.currentText() == 'CIRR':
            self.dataset = 'CIRR'
        return

    def choose_model(self):
        if self.comboBox_2.currentText() == 'fiq_mpac.pt':
            self.model = 'fiq_mpac.pt'
        elif self.comboBox_2.currentText() == 'cirr_mpac.pt':
            self.model = 'cirr_mpac.pt'
        elif self.comboBox_2.currentText() == 'pic2cup.pt':
            self.model = 'pic2cup.pt'
        return

    def choose_retrieval_mode(self):
        button_name = self.sender().objectName()
        n = re.findall(r'\d+', button_name)
        n = int(n[0])
        if n == 0:
            self.retrieval_mode = 't2i'
        elif n == 1:
            self.retrieval_mode = 'i2i'
        elif n == 2:
            self.retrieval_mode = 'it2i'
        elif n == 3:
            self.retrieval_mode = 'ii2i'
        print(self.retrieval_mode)
        return

    def adapt2win(self, view_w, view_h, img_w, img_h):
        ratio_w = view_w * 1.0 / img_w
        ratio_h = view_h * 1.0 / img_h
        if ratio_w > ratio_h:
            scale = ratio_h
        else:
            scale = ratio_w
            
        return scale

    def translate(self, ch):
        baidu_api = BaiduAPI()
        if bool(re.search(r'[\u4e00-\u9fff]', ch)):
            en = baidu_api.translate(ch)
        else:
            en = ch
        return en

    def clear_modificationText(self):
        self.textEdit.clear()
        self.textEdit_2.clear()

    def retrieval_and_show(self):
        if self.dataset == 'FashionIQ':
            cap0, cap1 = self.textEdit.toPlainText(), self.textEdit_2.toPlainText()
            cap = cap0 + ' & ' + cap1
            cap = self.translate(cap).lower()
            cap0, cap1 = cap.split('&')[0], cap.split('&')[1]
            cap = cap0 + ' & ' + cap1

            if 'dress' in cap.split(' ') or 'skirt' in cap.split(' '):
                self.style = 'dress'
            elif 'shirt' in cap.split(' '):
                self.style = 'shirt'
            elif 'top' in cap.split(' '):
                self.style = 'toptee'
            self.mod_text = [cap.split('&')[0].strip(), cap.split('&')[1].strip()]
        elif self.dataset == 'CIRR':
            self.mod_text = self.translate(self.textEdit.toPlainText())
        print(self.mod_text)

        model_type = self.model.split('.')[0]
        if model_type == 'fiq_mpac' or model_type == 'cirr_mpac':
            if self.ref_imgPath:
                dress_type = os.path.dirname(self.ref_imgPath).split("/")[-1]
                if dress_type in ['dress', 'shirt', 'toptee']:
                    self.style = dress_type
            print(self.style)
            print(self.retrieval_mode)
            res = inference_mpac(self.dataset, self.model_path, self.retrieval_mode, self.style,
                                 self.ref_imgPath, self.mod_text, self.ref_imgPath_list)
        elif model_type == 'pic2cup':
            res = inference_pic2cup(self.dataset, self.model_path, self.ref_imgPath, self.mod_text, self.retrieval_mode)

        self.res = res

        for i in range(len(res)):
            res_path = os.path.join(self.data_dir, res[i] + '.png')
            if res_path:
                image = QImage(res_path)
                if image.isNull():
                    QMessageBox.information(self, "Error", "Cannot load %s" % res_path)
                else:
                    pixmap = QPixmap.fromImage(image)
                    scale = self.adapt2win(self.graphicView[i].width(), self.graphicView[i].height(),
                                           pixmap.width(), pixmap.height())
                    item = QGraphicsPixmapItem(pixmap)
                    item.setScale(scale * 0.95)
                    # self.scene[i] = QGraphicsScene()
                    self.scene[i].clear()
                    self.scene[i].addItem(item)
                    self.graphicView[i].setScene(self.scene[i])
        self.reset_refImg_list()
        return

    def reset_refImg_list(self):
        self.ref_imgPath_list.clear()
        v = 'self.checkBox'
        for i in range(6):
            eval(v + '_' + str(i)).setChecked(False)
        return

    def clear_all(self):
        self.ref_scene.clear()
        for scene in self.scene:
            scene.clear()
        self.textEdit.clear()
        self.textEdit_2.clear()

class BaiduAPI():
    def __init__(self):
        endpoint = 'http://api.fanyi.baidu.com'
        path = '/api/trans/vip/translate'
        self.url = endpoint + path
        self.appid = '20240331002010226'
        self.appkey = '58Wi18TKorT61kICgWyx'

    def translate(self, text, from_lang='auto', to_lang='en'):
        # Generate salt and sign
        salt = random.randint(32768, 65536)
        sign = make_md5(self.appid + text + str(salt) + self.appkey)

        # Build request
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': self.appid, 'q': text, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

        # Send request
        r = requests.post(self.url, params=payload, headers=headers)
        result = r.json()

        # Show response
        # print(json.dumps(result, indent=4, ensure_ascii=False))
        return result['trans_result'][0]['dst']

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序对象
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())  # 在主线程中退出
