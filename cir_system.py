# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cir_system.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 471, 71))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(10, 20, 51, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(280, 20, 51, 31))
        self.label_2.setObjectName("label_2")
        self.comboBox = QtWidgets.QComboBox(self.frame)
        self.comboBox.setGeometry(QtCore.QRect(340, 20, 121, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(210, 20, 61, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit = QtWidgets.QLineEdit(self.frame)
        self.lineEdit.setGeometry(QtCore.QRect(50, 20, 151, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 100, 471, 551))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.graphicsView = QtWidgets.QGraphicsView(self.frame_2)
        self.graphicsView.setGeometry(QtCore.QRect(10, 40, 450, 450))
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.frame_2)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(9, 500, 451, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.frame_2)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(9, 0, 451, 31))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(10, 670, 471, 351))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.textEdit = QtWidgets.QTextEdit(self.frame_3)
        self.textEdit.setGeometry(QtCore.QRect(10, 50, 451, 131))
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.frame_3)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(9, 10, 451, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.textEdit_2 = QtWidgets.QTextEdit(self.frame_3)
        self.textEdit_2.setGeometry(QtCore.QRect(10, 200, 451, 131))
        self.textEdit_2.setObjectName("textEdit_2")
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(490, 10, 1421, 1011))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.frame_4)
        self.graphicsView_2.setGeometry(QtCore.QRect(20, 20, 450, 450))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_3.setGeometry(QtCore.QRect(660, 480, 111, 51))
        self.pushButton_3.setObjectName("pushButton_3")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.frame_4)
        self.graphicsView_3.setGeometry(QtCore.QRect(490, 20, 450, 450))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.graphicsView_4 = QtWidgets.QGraphicsView(self.frame_4)
        self.graphicsView_4.setGeometry(QtCore.QRect(960, 20, 450, 450))
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.graphicsView_5 = QtWidgets.QGraphicsView(self.frame_4)
        self.graphicsView_5.setGeometry(QtCore.QRect(20, 540, 450, 450))
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.graphicsView_6 = QtWidgets.QGraphicsView(self.frame_4)
        self.graphicsView_6.setGeometry(QtCore.QRect(490, 540, 450, 450))
        self.graphicsView_6.setObjectName("graphicsView_6")
        self.graphicsView_7 = QtWidgets.QGraphicsView(self.frame_4)
        self.graphicsView_7.setGeometry(QtCore.QRect(960, 540, 450, 450))
        self.graphicsView_7.setObjectName("graphicsView_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionOpen_2 = QtWidgets.QAction(MainWindow)
        self.actionOpen_2.setObjectName("actionOpen_2")
        self.actionSave_2 = QtWidgets.QAction(MainWindow)
        self.actionSave_2.setObjectName("actionSave_2")
        self.actionQuit_2 = QtWidgets.QAction(MainWindow)
        self.actionQuit_2.setObjectName("actionQuit_2")
        self.actionSetting = QtWidgets.QAction(MainWindow)
        self.actionSetting.setObjectName("actionSetting")
        self.menu.addAction(self.actionOpen_2)
        self.menu.addAction(self.actionSave_2)
        self.menu.addAction(self.actionSetting)
        self.menu.addAction(self.actionQuit_2)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(MainWindow.choose_refImage) # type: ignore
        self.pushButton_2.clicked.connect(MainWindow.load_model) # type: ignore
        self.pushButton_3.clicked.connect(MainWindow.retrieval_and_show) # type: ignore
        self.comboBox.activated['int'].connect(MainWindow.choose_dataset) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "模型"))
        self.label_2.setText(_translate("MainWindow", "数据集"))
        self.comboBox.setItemText(0, _translate("MainWindow", "FashionIQ"))
        self.comboBox.setItemText(1, _translate("MainWindow", "CIRR"))
        self.pushButton_2.setText(_translate("MainWindow", "浏览"))
        self.pushButton.setText(_translate("MainWindow", "浏览"))
        self.label_4.setText(_translate("MainWindow", "选择参考图像"))
        self.label_3.setText(_translate("MainWindow", "输入修改文本"))
        self.pushButton_3.setText(_translate("MainWindow", "开始检索"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "帮助"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionOpen_2.setText(_translate("MainWindow", "打开"))
        self.actionSave_2.setText(_translate("MainWindow", "保存"))
        self.actionQuit_2.setText(_translate("MainWindow", "退出"))
        self.actionSetting.setText(_translate("MainWindow", "设置"))