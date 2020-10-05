# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\MEGA\BioPhysics Lab\Optical System\gSTEDsFCS\ui\camerawindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Camera(object):
    def setupUi(self, Camera):
        Camera.setObjectName("Camera")
        Camera.resize(550, 580)
        Camera.setMinimumSize(QtCore.QSize(550, 580))
        Camera.setMaximumSize(QtCore.QSize(9999, 9999))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/devices/camera-photo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Camera.setWindowIcon(icon)
        self.gridLayout_2 = QtWidgets.QGridLayout(Camera)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(Camera)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(528, 558))
        self.frame.setObjectName("frame")
        self.gridLayout_34 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_34.setObjectName("gridLayout_34")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.shootButton = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.shootButton.sizePolicy().hasHeightForWidth())
        self.shootButton.setSizePolicy(sizePolicy)
        self.shootButton.setAutoDefault(False)
        self.shootButton.setDefault(False)
        self.shootButton.setFlat(False)
        self.shootButton.setObjectName("shootButton")
        self.verticalLayout_5.addWidget(self.shootButton)
        self.videoButton = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.videoButton.sizePolicy().hasHeightForWidth())
        self.videoButton.setSizePolicy(sizePolicy)
        self.videoButton.setMinimumSize(QtCore.QSize(0, 0))
        self.videoButton.setObjectName("videoButton")
        self.verticalLayout_5.addWidget(self.videoButton)
        self.gridLayout.addLayout(self.verticalLayout_5, 0, 0, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalLayout_36 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_36.setSpacing(3)
        self.horizontalLayout_36.setObjectName("horizontalLayout_36")
        self.verticalLayout_63 = QtWidgets.QVBoxLayout()
        self.verticalLayout_63.setObjectName("verticalLayout_63")
        self.label_64 = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_64.sizePolicy().hasHeightForWidth())
        self.label_64.setSizePolicy(sizePolicy)
        self.label_64.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_64.setFont(font)
        self.label_64.setAlignment(QtCore.Qt.AlignCenter)
        self.label_64.setObjectName("label_64")
        self.verticalLayout_63.addWidget(self.label_64)
        self.contrastSlider = QtWidgets.QSlider(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.contrastSlider.sizePolicy().hasHeightForWidth())
        self.contrastSlider.setSizePolicy(sizePolicy)
        self.contrastSlider.setSingleStep(1)
        self.contrastSlider.setPageStep(10)
        self.contrastSlider.setProperty("value", 70)
        self.contrastSlider.setOrientation(QtCore.Qt.Vertical)
        self.contrastSlider.setObjectName("contrastSlider")
        self.verticalLayout_63.addWidget(self.contrastSlider, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout_63.setStretch(1, 1)
        self.horizontalLayout_36.addLayout(self.verticalLayout_63)
        self.verticalLayout_64 = QtWidgets.QVBoxLayout()
        self.verticalLayout_64.setObjectName("verticalLayout_64")
        self.label_65 = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_65.sizePolicy().hasHeightForWidth())
        self.label_65.setSizePolicy(sizePolicy)
        self.label_65.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_65.setFont(font)
        self.label_65.setAlignment(QtCore.Qt.AlignCenter)
        self.label_65.setObjectName("label_65")
        self.verticalLayout_64.addWidget(self.label_65)
        self.brightnessSlider = QtWidgets.QSlider(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.brightnessSlider.sizePolicy().hasHeightForWidth())
        self.brightnessSlider.setSizePolicy(sizePolicy)
        self.brightnessSlider.setProperty("value", 70)
        self.brightnessSlider.setOrientation(QtCore.Qt.Vertical)
        self.brightnessSlider.setObjectName("brightnessSlider")
        self.verticalLayout_64.addWidget(self.brightnessSlider, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout_64.setStretch(1, 1)
        self.horizontalLayout_36.addLayout(self.verticalLayout_64)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_36)
        self.viewToolsGroup = QtWidgets.QGroupBox(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.viewToolsGroup.sizePolicy().hasHeightForWidth())
        self.viewToolsGroup.setSizePolicy(sizePolicy)
        self.viewToolsGroup.setObjectName("viewToolsGroup")
        self.magToolButton = QtWidgets.QToolButton(self.viewToolsGroup)
        self.magToolButton.setGeometry(QtCore.QRect(50, 20, 32, 32))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.magToolButton.sizePolicy().hasHeightForWidth())
        self.magToolButton.setSizePolicy(sizePolicy)
        self.magToolButton.setMinimumSize(QtCore.QSize(32, 32))
        self.magToolButton.setMaximumSize(QtCore.QSize(32, 32))
        self.magToolButton.setStyleSheet("background-image: url(:/icons/actions/system-search.png);\n"
"background-position: center;\n"
"background-repeat: no-repeat;")
        self.magToolButton.setText("")
        self.magToolButton.setCheckable(True)
        self.magToolButton.setAutoExclusive(True)
        self.magToolButton.setObjectName("magToolButton")
        self.ROItoolButton = QtWidgets.QToolButton(self.viewToolsGroup)
        self.ROItoolButton.setGeometry(QtCore.QRect(10, 20, 32, 32))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ROItoolButton.sizePolicy().hasHeightForWidth())
        self.ROItoolButton.setSizePolicy(sizePolicy)
        self.ROItoolButton.setMinimumSize(QtCore.QSize(32, 32))
        self.ROItoolButton.setMaximumSize(QtCore.QSize(32, 32))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.ROItoolButton.setFont(font)
        self.ROItoolButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.ROItoolButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/myIcons/crosshair.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ROItoolButton.setIcon(icon1)
        self.ROItoolButton.setIconSize(QtCore.QSize(25, 25))
        self.ROItoolButton.setCheckable(True)
        self.ROItoolButton.setAutoExclusive(True)
        self.ROItoolButton.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.ROItoolButton.setObjectName("ROItoolButton")
        self.horizontalLayout_7.addWidget(self.viewToolsGroup)
        self.horizontalLayout_7.setStretch(1, 1)
        self.gridLayout.addLayout(self.horizontalLayout_7, 1, 1, 1, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout_34.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Camera)
        QtCore.QMetaObject.connectSlotsByName(Camera)

    def retranslateUi(self, Camera):
        _translate = QtCore.QCoreApplication.translate
        Camera.setWindowTitle(_translate("Camera", "Camera"))
        self.shootButton.setText(_translate("Camera", "Take Photo"))
        self.videoButton.setText(_translate("Camera", "Start Video"))
        self.label_64.setText(_translate("Camera", "Contrast"))
        self.label_65.setText(_translate("Camera", "Brightness"))
        self.viewToolsGroup.setTitle(_translate("Camera", "Tools"))
        self.magToolButton.setToolTip(_translate("Camera", "magnify"))
        self.ROItoolButton.setToolTip(_translate("Camera", "crosshair"))
import icons_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Camera = QtWidgets.QWidget()
    ui = Ui_Camera()
    ui.setupUi(Camera)
    Camera.show()
    sys.exit(app.exec_())
