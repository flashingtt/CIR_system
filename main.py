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
        self.ui_init()

    def ui_init(self):

        # self.textEdit.setStyleSheet(
        #     '''border: 1px solid black;
        #        border-radius: 6px;'''
        # )
        # self.plainTextEdit.setStyleSheet(
        #     '''border: 2px solid black;
        #     border-radius: 10px;
        #     '''
        # )

        # self.label.setStyleSheet(
        #     '''font-family: 微软雅黑'''
        # )

        # self.pushButton_3.setStyleSheet(
        #     '''border: 2px solid black;
        #     border-radius: 10px'''
        # )

        # init reference display module
        v = 'self.refButton_'
        for i in range(4):
            eval(v + str(i)).setStyleSheet(
            '''QPushButton{color: rgba(0, 0, 0, 0);
                           background-color: rgba(255, 132, 139, 0);
                           border-style: groove;}
               QPushButton:hover{color: rgba(0, 0, 0, 0.5);
                                 font-family: 微软雅黑;
                                 font-size: 25px;
                                 letter-spacing: 2px;
                                 background-color: rgba(224, 238, 249, 0.5);
                                 border-style: groove;}
               QPushButton:pressed{color: rgba(0, 0, 0, 1);
                                   background-color: rgba(204, 228, 247, 0.5);
                                   border-style: groove}''')

        # init result display module
        u = 'self.pushButton_resetRef'
        for i in range(12):
            eval(u + '_' + str(i)).setStyleSheet(
            '''QPushButton{color: rgba(0, 0, 0, 0);
                           background-color: rgba(255, 132, 139, 0);
                           border-style: groove;}
               QPushButton:hover{color: rgba(0, 0, 0, 0.5);
                                 font-family: 微软雅黑;
                                 font-size: 25px;
                                 letter-spacing: 2px;
                                 background-color: rgba(224, 238, 249, 0.5);
                                 border-style: groove;}
               QPushButton:pressed{color: rgba(0, 0, 0, 1);
                                   background-color: rgba(204, 228, 247, 0.5);
                                   border-style: groove}''')

        # init path
        self.database_dir = 'database'
        self.data_dir = 'database/fashionIQ/images'
        self.dataset = 'FashionIQ'
        self.model_path = 'models/fiq_mpac.pt'
        self.model = 'fiq_mpac.pt'

        self.save_dir = 'results'

        # init retrieval mode
        modeid = self.tabWidget.currentIndex()
        if modeid == 0:
            self.retrieval_mode = 't2i'
        elif modeid == 1:
            self.retrieval_mode = 'i2i'
        elif modeid == 2:
            self.retrieval_mode = 'it2i'
        else:
            self.retrieval_mode = 'ii2i'

        # init visualization module
        self.ref_scene = []
        # u = 'self.ref_scene'
        for i in range(4):
            self.ref_scene.append(QGraphicsScene())

        # input
        self.ref_imgPath = None
        self.ref_imgPath_1 = None
        self.mod_text = None
        self.ref_imgPath_list = []
        self.ii2i_ref_0 = None
        self.ii2i_ref_1 = None

        # init cap
        self.cap, self.cap0, self.cap1 = None, None, None

        # init visualization module
        # self.ref_scene = QGraphicsScene()

        self.scene = []
        self.graphicView = []
        v = 'self.graphicsView'
        for i in range(12):
            self.scene.append(QGraphicsScene())
            self.graphicView.append(eval(v + '_' + str(i)))
        # self.graphicView = [self.graphicsView_0, self.graphicsView_1, self.graphicsView_2,
        #                     self.graphicsView_3, self.graphicsView_4, self.graphicsView_5]

        # init other details
        self.res = None
        self.style = None
        self.style_signal = None

    # def show_refImage(self, ref_imgPath):
    #     if ref_imgPath:
    #         ref_image = QImage(ref_imgPath)
    #         if ref_image.isNull():
    #             QMessageBox.information(self, "Error", "Cannot load %s." % ref_imgPath)
    #         else:
    #             pixmap = QPixmap.fromImage(ref_image)
    #             scale = self.adapt2win(self.graphicsView.width(), self.graphicsView.height(),
    #                                    pixmap.width(), pixmap.height())
    #             item = QGraphicsPixmapItem(pixmap)
    #             item.setScale(scale * 0.95)
    #             # scene = QGraphicsScene()
    #             self.ref_scene.clear()
    #             self.ref_scene.addItem(item)
    #             self.graphicsView.setScene(self.ref_scene)
    #     self.ref_imgPath = ref_imgPath
    #     return

    def show_refImage(self, ref_imgPath, graphicsView, ref_scene):
        if ref_imgPath:
            ref_image = QImage(ref_imgPath)
            if ref_image.isNull():
                QMessageBox.information(self, "Error", "Cannot load %s." % ref_imgPath)
            else:
                pixmap = QPixmap.fromImage(ref_image)
                scale = self.adapt2win(graphicsView.width(), graphicsView.height(),
                                       pixmap.width(), pixmap.height())
                item = QGraphicsPixmapItem(pixmap)
                item.setScale(scale * 0.95)
                # scene = QGraphicsScene()
                ref_scene.clear()
                ref_scene.addItem(item)
                graphicsView.setScene(ref_scene)
        self.ref_imgPath = ref_imgPath
        return

    def save_image(self):
        button_name = self.sender().objectName()
        n = re.findall(r'\d+', button_name)
        n = int(n[0])
        img_path = os.path.join(self.data_dir, self.res[n] + '.png')
        base_name = os.path.basename(img_path)
        save_path = os.path.join(self.save_dir, base_name)
        save_image = Image.open(img_path)
        save_image.save(save_path)
        print("Save success!")
        return

    def get_save_dir(self):
        self.save_dir = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "")
        self.plainTextEdit.setPlainText(self.save_dir)
        return

    def choose_refImage(self):
        ref_imgPath, _ = QFileDialog.getOpenFileName(self, "Open Pictures", "", "Image (*.jpg *.png)")
        button_name = self.sender().objectName()
        n = re.findall(r'\d+', button_name)
        n = int(n[0])
        self.show_refImage(ref_imgPath, eval('self.refView' + '_' + str(n)), self.ref_scene[n])
        return

    def reset_refImg(self):
        button_name = self.sender().objectName()
        n = re.findall(r'\d+', button_name)
        n = int(n[0])
        img_path = os.path.join(self.data_dir, self.res[n] + '.png')

        if self.retrieval_mode == 'i2i':
            self.ref_scene[0].clear()
            self.show_refImage(img_path, eval('self.refView_' + str(0)), self.ref_scene[0])
        elif self.retrieval_mode == 'it2i':
            self.ref_scene[1].clear()
            self.show_refImage(img_path, eval('self.refView_' + str(1)), self.ref_scene[1])
        elif self.retrieval_mode == 'ii2i':
            if self.radioButton_refii2i_0.isChecked():
                self.ref_scene[2].clear()
                self.show_refImage(img_path, eval('self.refView_' + str(2)), self.ref_scene[2])
                self.ii2i_ref_0 = img_path
            elif self.radioButton_refii2i_1.isChecked():
                self.ref_scene[3].clear()
                self.show_refImage(img_path, eval('self.refView_' + str(3)), self.ref_scene[3])
                self.ii2i_ref_1 = img_path
            else:
                self.ref_scene[2].clear()
                self.show_refImage(img_path, eval('self.refView_' + str(2)), self.ref_scene[2])
                self.ii2i_ref_0 = img_path
        else:
            pass
        return

    def add_refImgs(self):
        button_name = self.sender().objectName()
        n = re.findall(r'\d+', button_name)
        n = int(n[0])
        img_path = os.path.join(self.data_dir, self.res[n] + '.png')

        if self.sender().isChecked():
            self.ref_imgPath_list.append(img_path)
        else:
            if self.ref_imgPath_list is not None:
                self.ref_imgPath_list.remove(img_path)
        print(self.ref_imgPath_list)

        if self.radioButton_refii2i_0.isChecked():
            self.ref_scene[2].clear()
            self.show_refImage(img_path, eval('self.refView_' + str(2)), self.ref_scene[2])
            # self.ii2i_ref_0 = img_path
        elif self.radioButton_refii2i_1.isChecked():
            self.ref_scene[3].clear()
            self.show_refImage(img_path, eval('self.refView_' + str(3)), self.ref_scene[3])
            # self.ii2i_ref_1 = img_path
        else:
            self.ref_scene[2].clear()
            self.show_refImage(img_path, eval('self.refView_' + str(2)), self.ref_scene[2])
            # self.ii2i_ref_0 = img_path
        return

    def load_model(self):
        # model_path, _ = QFileDialog.getOpenFileName(self, "Open Pictures", "", "Model (*.pt)")
        model_dir = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "")
        self.model_path = os.path.join(model_dir, self.model)
        # self.lineEdit.setText(os.path.basename(model_path))
        return

    def load_data_dir(self):
        self.database_dir = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "")
        return

    def choose_dataset(self):
        if self.comboBox.currentText() == 'FashionIQ':
            self.dataset = 'FashionIQ'
            self.data_dir = os.path.join(self.database_dir, 'fashionIQ/images')
        elif self.comboBox.currentText() == 'CIRR':
            self.dataset = 'CIRR'
            self.data_dir = os.path.join(self.database_dir, 'cirr/dev')
        print(self.data_dir)
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
        # button_name = self.sender().objectName()
        # n = re.findall(r'\d+', button_name)
        # n = int(n[0])
        n = self.tabWidget.currentIndex()
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
            if self.retrieval_mode == 't2i':
                cap0 = self.textEdit.toPlainText()
                cap0 = self.translate(cap0).lower()
                cap = cap0
            elif self.retrieval_mode == 'it2i':
                cap1 = self.textEdit_2.toPlainText()
                cap1 = self.translate(cap1).lower()
                cap = cap1
            else:
                cap = self.style

            if 'dress' in cap or 'skirt' in cap:
                self.style = 'dress'
            elif 'shirt' in cap:
                self.style = 'shirt'
            elif 'top' in cap:
                self.style = 'toptee'

            self.mod_text = [cap, '']

        elif self.dataset == 'CIRR':
            if self.retrieval_mode == 't2i':
                self.mod_text = self.translate(self.textEdit.toPlainText())
            elif self.retrieval_mode == 'it2i':
                self.mod_text = self.translate(self.textEdit_2.toPlainText())
        print(self.mod_text)

        model_type = self.model.split('.')[0]
        # for ii2i mode
        # if self.ii2i_ref_0 is not None:
        #     self.ref_imgPath_list.append(self.ii2i_ref_0)
        # if self.ii2i_ref_1 is not None:
        #     self.ref_imgPath_list.append(self.ii2i_ref_1)
        # print(self.ref_imgPath_list)
        if model_type == 'fiq_mpac' or model_type == 'cirr_mpac':
            if self.ref_imgPath:
                dress_type = os.path.dirname(self.ref_imgPath).split("/")[-1]
                if dress_type in ['dress', 'shirt', 'toptee']:
                    self.style = dress_type
            print(self.style)
            res = inference_mpac(self.dataset, self.model_path, self.retrieval_mode, self.style,
                                 self.ref_imgPath, self.mod_text, self.ref_imgPath_list)
        elif model_type == 'pic2cup':
            res = inference_pic2cup(self.dataset, self.model_path, self.retrieval_mode, 
                                    self.ref_imgPath, self.mod_text, self.ref_imgPath_list)

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
        for scene in self.ref_scene:
            scene.clear()
        # self.ref_scene.clear()
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
