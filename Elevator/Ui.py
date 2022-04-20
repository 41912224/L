from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import *
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLCDNumber
import sys
from Dispatcher import Dispatcher

# 定义窗口界面：
class Window(QWidget):

    def __init__(self):
        # 定义背景图
        super().__init__()
        # self.center()
        # 定义尺寸大小
        self.resize(650, 1000)
        # 距离边框偏离大小
        self.move(600, 0)
        palette1 = QtGui.QPalette()
        palette1.setColor(self.backgroundRole(), QColor(255, 255, 255))
        self.setPalette(palette1)
        # 导入背景图片
        self.setObjectName("window")
        self.setStyleSheet("#window{border-image:url(D:/icon/background.png)}")

        # 定义电梯类
        self.lift = {}
        for i in range(1, 3):
            self.lift[i] = QtWidgets.QLabel(self)  # 以window为父控件
            self.lift[i].setPixmap(QtGui.QPixmap("D:/icon/lift.png"))  # 导入电梯图片
            self.lift[i].setGeometry(QtCore.QRect(i * 190 - 180, 955, 36, 36))  # 定义尺寸大小
            self.lift[i].setScaledContents(True)
        self.lift_anime = {}
        for i in range(1, 3):
            self.lift_anime[i] = QtCore.QPropertyAnimation(self.lift[i], b"geometry")
        # 放置校徽
        self.lif = QtWidgets.QLabel(self)  # 以window为父控件
        self.lif.setPixmap(QtGui.QPixmap("D:/icon/ground.png"))
        self.lif.resize(200, 200)  # 校徽尺寸
        self.lif.setGeometry(QtCore.QRect(500, 0, 100, 100))  # 校徽位置

        # 程序框上部的数字显示
        self.floor_digit = {}
        for i in range(1, 3):
            self.floor_digit[i] = QLCDNumber(2, self)
            self.floor_digit[i].setGeometry(QtCore.QRect(i * 190 - 180, 0, 100, 100))  # 位置
            self.floor_digit[i].resize(100, 100)  # 定义尺寸大小
            self.floor_digit[i].display('2')  # 顶部数字初始值
        # 定义电梯按钮
        self.up_btn = {}
        self.down_btn = {}
        # 导入图片
        self.up_btn_md = "QPushButton{border-image: url(D:/icon/up_hover.png)}" \
                       "QPushButton:hover{border-image: url(D:/icon/up.png)}" \
                       "QPushButton:pressed{border-image: url(D:/icon/up_pressed.png)}"
        self.down_btn_md = "QPushButton{border-image: url(D:/icon/down_hover.png)}" \
                         "QPushButton:hover{border-image: url(D:/icon/down.png)}" \
                         "QPushButton:pressed{border-image: url(D:/icon/down_pressed.png)}"
        for i in range(1, 20):
            self.up_btn[i] = QtWidgets.QPushButton(self)
            self.up_btn[i].setGeometry(QtCore.QRect(500, 1000 - i * 45, 40, 40))  # 不同按钮位置
            self.up_btn[i].setStyleSheet(self.up_btn_md)
        for i in range(2, 21):
            self.down_btn[i] = QtWidgets.QPushButton(self)
            self.down_btn[i].setGeometry(QtCore.QRect(550, 1000 - i * 45, 40, 40))  # 不同按钮位置
            self.down_btn[i].setStyleSheet(self.down_btn_md)

        self.floor_btn = [[] for i in range(6)]
        for i in range(1, 3):
            self.floor_btn[i].append("Nothing")
            for j in range(1, 21):
                self.floor_btn[i].append(QtWidgets.QPushButton(self))  # 创建一个按钮，并将按钮加入到窗口MainWindow中
                self.floor_btn[i][j].setGeometry(QtCore.QRect(i * 190 - 130, 1000 - j * 45, 40, 40))  # 位置大小
                # 导入图片
                self.floor_btn[i][j].setStyleSheet(
                    "QPushButton{border-image: url(D:/icon/" + str(j) + "_hover.png)}"
                    "QPushButton:hover{border-image: url(D:/icon/" + str(j) + ".png)}"
                    "QPushButton:pressed{border-image: url(D:/icon/" + str(j) + "_pressed.png)}")
        self.dispatcher = Dispatcher(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
