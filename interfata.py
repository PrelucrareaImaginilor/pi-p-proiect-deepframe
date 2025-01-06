# -*- coding: utf-8 -*-
import os
import sys
import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage
from ultralytics import YOLO


class FrameSequenceThread(QThread):
    frame_ready = pyqtSignal(object, list)  # Modified to also emit detection results

    def __init__(self, frames_dir, model, threshold):
        super().__init__()
        self.frames_dir = frames_dir
        self.model = model
        self.threshold = threshold
        self.running = False
        self.detect_objects = False
        self.frame_files = sorted([f for f in os.listdir(frames_dir)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def run(self):
        self.running = True
        for frame_file in self.frame_files:
            if not self.running:
                break
            frame = cv.imread(os.path.join(self.frames_dir, frame_file))

            if frame is None:
                print(f"Eroare la încărcarea frame-ului: {frame_file}")
                continue

            # Aplicăm detecția doar dacă este activată
            if self.detect_objects:
                results = self.model.predict(frame, conf=self.threshold / 100.0)
                self.frame_ready.emit(frame, results)  # Emitem frame-ul și detecțiile
            else:
                self.frame_ready.emit(frame, [])  # Dacă nu, doar frame-ul

            QtCore.QThread.msleep(33)  # ~30 FPS

    def stop(self):
        self.running = False


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(955, 587)
        self.MainWindow=MainWindow
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(40, 460, 661, 21))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.Imagine = QtWidgets.QLabel(self.centralwidget)
        self.Imagine.setGeometry(QtCore.QRect(20, 10, 681, 441))
        self.Imagine.setObjectName("Imagine")
        self.DetectButton = QtWidgets.QPushButton(self.centralwidget)
        self.DetectButton.setGeometry(QtCore.QRect(780, 430, 111, 31))
        self.DetectButton.setObjectName("DetectButton")
        self.batchButton = QtWidgets.QPushButton(self.centralwidget)
        self.batchButton.setGeometry(QtCore.QRect(780, 480, 111, 31))
        self.batchButton.setObjectName("batchButton")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(780, 20, 141, 291))
        self.groupBox.setObjectName("groupBox")
        self.widget = QtWidgets.QWidget(self.groupBox)
        self.widget.setGeometry(QtCore.QRect(10, 20, 99, 255))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.checkBox = QtWidgets.QCheckBox(self.widget)
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout.addWidget(self.checkBox)
        self.checkBox_2 = QtWidgets.QCheckBox(self.widget)
        self.checkBox_2.setChecked(True)
        self.checkBox_2.setObjectName("checkBox_2")
        self.verticalLayout.addWidget(self.checkBox_2)
        self.checkBox_3 = QtWidgets.QCheckBox(self.widget)
        self.checkBox_3.setChecked(True)
        self.checkBox_3.setObjectName("checkBox_3")
        self.verticalLayout.addWidget(self.checkBox_3)
        self.checkBox_4 = QtWidgets.QCheckBox(self.widget)
        self.checkBox_4.setChecked(True)
        self.checkBox_4.setObjectName("checkBox_4")
        self.verticalLayout.addWidget(self.checkBox_4)
        self.checkBox_5 = QtWidgets.QCheckBox(self.widget)
        self.checkBox_5.setChecked(True)
        self.checkBox_5.setObjectName("checkBox_5")
        self.verticalLayout.addWidget(self.checkBox_5)
        self.checkBox_6 = QtWidgets.QCheckBox(self.widget)
        self.checkBox_6.setChecked(True)
        self.checkBox_6.setObjectName("checkBox_6")
        self.verticalLayout.addWidget(self.checkBox_6)
        self.checkBox_7 = QtWidgets.QCheckBox(self.widget)
        self.checkBox_7.setChecked(True)
        self.checkBox_7.setObjectName("checkBox_7")
        self.verticalLayout.addWidget(self.checkBox_7)
        self.checkBox_8 = QtWidgets.QCheckBox(self.widget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.checkBox_8.setFont(font)
        self.checkBox_8.setChecked(True)
        self.checkBox_8.setObjectName("checkBox_8")
        self.verticalLayout.addWidget(self.checkBox_8)
        self.checkBox_9 = QtWidgets.QCheckBox(self.widget)
        self.checkBox_9.setChecked(True)
        self.checkBox_9.setObjectName("checkBox_9")
        self.verticalLayout.addWidget(self.checkBox_9)
        self.checkBox_10 = QtWidgets.QCheckBox(self.widget)
        self.checkBox_10.setChecked(True)
        self.checkBox_10.setObjectName("checkBox_10")
        self.verticalLayout.addWidget(self.checkBox_10)
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(760, 380, 160, 18))
        self.horizontalSlider.setMaximum(100)
        self.horizontalSlider.setSliderPosition(50)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(750, 350, 131, 16))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 955, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNou = QtWidgets.QAction(MainWindow)
        self.actionNou.setObjectName("actionNou")
        self.actionSterge = QtWidgets.QAction(MainWindow)
        self.actionSterge.setObjectName("actionSterge")
        self.actionSecven_video = QtWidgets.QAction(MainWindow)
        self.actionSecven_video.setObjectName("actionSecven_video")
        self.menuFile.addAction(self.actionNou)
        self.menuFile.addAction(self.actionSterge)
        self.menuFile.addAction(self.actionSecven_video)
        self.menubar.addAction(self.menuFile.menuAction())

        # Aici se conecteaza actiunile la sloturi
        self.actionNou.triggered.connect(self.loadImage)  # Conectează la metoda loadImage
        self.actionSterge.triggered.connect(self.deletePhoto)
        self.DetectButton.clicked.connect(self.start_detection)
        self.actionSecven_video.triggered.connect(self.action_load_sequence)
        self.retranslateUi(MainWindow)
        self.menubar.customContextMenuRequested['QPoint'].connect(self.Imagine.clear)  # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Initialize additional variables
        self.frames_dir = None
        self.video_thread = None
        self.current_frame = None
        self.current_results = None

        # Load YOLO model
        try:
            self.model = YOLO("res/best.pt")
        except Exception as e:
            QMessageBox.critical(self.MainWindow, "Eroare", f"Nu s-a putut încărca modelul YOLO: {e}")
            sys.exit(1)



        # Connect all checkboxes to update_display
        self.checkboxes = [
            self.checkBox, self.checkBox_2, self.checkBox_3, self.checkBox_4,
            self.checkBox_5, self.checkBox_6, self.checkBox_7, self.checkBox_8,
            self.checkBox_9, self.checkBox_10
        ]

        # Map checkboxes to class names
        self.class_map = {
            self.checkBox: "pedestrian",
            self.checkBox_2: "people",
            self.checkBox_3: "bicycle",
            self.checkBox_4: "car",
            self.checkBox_5: "van",
            self.checkBox_6: "truck",
            self.checkBox_7: "tricycle",
            self.checkBox_8: "awning-tricycle",
            self.checkBox_9: "bus",
            self.checkBox_10: "motor"
        }

        for checkbox in self.checkboxes:
            checkbox.stateChanged.connect(self.update_display)

    def update_threshold(self):
        detection_threshold = self.horizontalSlider.value()

        # Dacă există o imagine, relansăm detecția
        if hasattr(self, 'image') and self.image is not None:
            results = self.model.predict(self.image, conf=detection_threshold / 100.0)
            self.process_frame(self.image, results)

        # Dacă secvența video rulează, actualizăm pragul pentru viitoarele frame-uri
        if self.video_thread:
            self.video_thread.threshold = detection_threshold

    def action_load_sequence(self):
        self.frames_dir = QFileDialog.getExistingDirectory(self.MainWindow, "Select Frames Directory")
        if self.frames_dir:
            self.start_frame_sequence()

    def start_frame_sequence(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()

        detection_threshold = self.horizontalSlider.value()
        self.video_thread = FrameSequenceThread(self.frames_dir, self.model, detection_threshold)
        self.video_thread.frame_ready.connect(self.process_frame)
        self.video_thread.start()

    def start_detection(self):
        detection_threshold = self.horizontalSlider.value()

        # Dacă avem o secvență video, actualizăm pragul și activăm detecția
        if self.video_thread and self.video_thread.running:
            self.video_thread.detect_objects = True
            self.video_thread.threshold = detection_threshold

        # Dacă avem o imagine statică, efectuăm detecția imediat
        elif hasattr(self, 'image') and self.image is not None:
            results = self.model.predict(self.image, conf=detection_threshold / 100.0)
            self.process_frame(self.image, results)
        else:
            QMessageBox.warning(self.MainWindow, "Eroare",
                                "Încarcă o imagine sau pornește secvența video înainte de detectare.")

    def process_frame(self, frame, results):
        # Actualizăm frame-ul curent și rezultatele
        self.current_frame = frame.copy()
        self.current_results = results if results else []

        # Apelăm afișarea pentru a curăța și desena detecțiile
        self.update_display()

    def update_display(self):
        if self.current_frame is None:
            return

        # Lucrăm cu o copie curată a frame-ului
        display_frame = self.current_frame.copy()

        # Dacă avem rezultate de detecție, le procesăm
        if self.current_results:
            selected_classes = [self.class_map[cb] for cb in self.checkboxes if cb.isChecked()]

            for result in self.current_results:
                boxes = result.boxes
                for box in boxes:
                    cls = result.names[int(box.cls[0])]
                    if cls in selected_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])

                        # Desenăm doar obiectele selectate
                        cv.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f'{cls} {conf:.2f}'
                        cv.putText(display_frame, label, (int(x1), int(y1) - 10),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Afișăm frame-ul actualizat
        display_frame = cv.resize(display_frame, (681, 441))
        rgb_image = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                          rgb_image.strides[0], QImage.Format_RGB888)
        self.Imagine.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def loadImage(self):

        self.filename = QFileDialog.getOpenFileName(directory="G:/__PI Proiect/pythonProject1/res/images",filter="Image (*.png *.jpg *.jpeg *.bmp)")[0]
        #self.defaultValues()
        if self.filename:
            self.image = cv.imread(self.filename)
            if self.image is None:
                QMessageBox.warning(self.MainWindow, "Eroare",
                                    "Imaginea nu a putut fi încărcată. Verifică dacă fișierul este valid.")
                return

            self.tmp = self.image.copy()
            self.setPhoto(self.image)

    def setPhoto(self, image):
        image = cv.resize(image, [681, 441], interpolation=cv.INTER_AREA)
        frame = cv.cvtColor(image, cv.IMREAD_ANYCOLOR)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)

        self.Imagine.setPixmap(QtGui.QPixmap.fromImage(image))

    def deletePhoto(self):
        self.Imagine.setText("Încarcă o imagine pentru a începe.")

    def displayImage(self, img):
        """Afișează imaginea în QLabel după redimensionare."""
        img = cv.resize(img, (681, 441))
        rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0],
                          QImage.Format_RGB888)
        self.Imagine.setPixmap(QtGui.QPixmap.fromImage(qt_image))
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DeepFrame"))
        self.Imagine.setText(_translate("MainWindow", "Încărcați o imagine sau un video"))
        self.DetectButton.setText(_translate("MainWindow", "Detecție obiecte"))
        self.batchButton.setText(_translate("MainWindow", "Procesare batch"))
        self.groupBox.setTitle(_translate("MainWindow", "Clase"))
        self.checkBox.setText(_translate("MainWindow", "pedestrian"))
        self.checkBox_2.setText(_translate("MainWindow", "people"))
        self.checkBox_3.setText(_translate("MainWindow", "bicycle"))
        self.checkBox_4.setText(_translate("MainWindow", "car"))
        self.checkBox_5.setText(_translate("MainWindow", "van"))
        self.checkBox_6.setText(_translate("MainWindow", "truck"))
        self.checkBox_7.setText(_translate("MainWindow", "tricycle"))
        self.checkBox_8.setText(_translate("MainWindow", "awning-tricycle"))
        self.checkBox_9.setText(_translate("MainWindow", "bus"))
        self.checkBox_10.setText(_translate("MainWindow", "motor"))
        self.label.setText(_translate("MainWindow", "Prag de încredere"))
        self.menuFile.setTitle(_translate("MainWindow", "Fișier"))
        self.actionNou.setText(_translate("MainWindow", "Nou"))
        self.actionSterge.setText(_translate("MainWindow", "Șterge fișierul"))
        self.actionSecven_video.setText(_translate("MainWindow", "Secvență video"))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
