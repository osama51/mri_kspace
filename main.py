import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
from PIL import Image
from PyQt5.uic import loadUiType
from PyQt5 import QtCore, QtGui
from pyqtgraph import ImageView
from PyQt5.QtGui import QBrush, QColor
from os import path
import cv2
import qdarktheme
import json
import matplotlib.pyplot as plt

from kspace import KSpace
from circle_item import CircleItem
import sequence
import phantom


FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui_mri.ui"))
class MriMain(QtWidgets.QMainWindow, FORM_CLASS):

    def __init__(self):
        pg.setConfigOptions(imageAxisOrder='row-major') 
        # Interpret image data as row-major instead of col-major
        pg.setConfigOption('background', (15,20, 20))
        super(MriMain,self).__init__()

        # Basic UI layout
        self.setupUi(self)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)
        self.glw = pg.GraphicsLayoutWidget()
        # self.setCentralWidget(self.glw)
        
        self.file_path_name = ""
        
        self.weights = ["PD", "T1", "T2"]
        self.selected_weight = self.weights[0]
        self.pdRadioButton.toggled.connect(self.select_pd)
        self.t1RadioButton.toggled.connect(self.select_t1)
        self.t2RadioButton.toggled.connect(self.select_t2)
        
        self.selected_size = 0
        self.sizeComboBox.currentIndexChanged.connect(self.select_size)
        
        self.reconstDefaultSize.clicked.connect(lambda: self.reconstructedView.autoRange())
        self.phantomDefaultSize.clicked.connect(lambda: [self.phantomView.autoRange(), self.kspaceView.autoRange()])
        self.actionReconstruct.triggered.connect(lambda: [self.start_kspace(self.kspaceView, self.reconstructedView)])
        self.actionStop.triggered.connect(self.stop_timer)
        self.actionOpen.triggered.connect(self.browse)
        # self.markerCheckBox.stateChanged.connect(self.set_point)
        
        self.rfDial.valueChanged.connect(self.sequence)
        
        
        # self.plot = self.phantomView.addPlot()
        self.x_i = 0
        self.y_i = 0
        
        self.counter = 0
        self.interval = 2000
        # self.draw_graph(self.plot)
        
        self.imageViews = [self.reconstructedView, self.phantomView, self.kspaceView]
        self.hideHisto()
        
        self.select_size(self.selected_size)  
        self.sequence()

        # Connect the mouseClicked signal to the custom slot
        # self.reconstructedView.mouseClicked.connect(self.handle_mouse_clicked)
    
    def select_size(self, index):
        self.selected_size = index
        if(index == 0):
            self.file_path_name = "weight/brain16_4.png"
        elif(index == 1):
            self.file_path_name = "weight/brain32_4.png"
        elif(index == 2):
            self.file_path_name = "weight/brain64_4.png"
        
        self.phantom_img = cv2.imread(self.file_path_name,0)
        self.phantom_img = np.array(self.phantom_img)
        
        self.num_of_rows = self.phantom_img.shape[0]
        self.num_of_cols = self.phantom_img.shape[1]
        self.k_space = np.ones((self.num_of_rows, self.num_of_cols), dtype=np.complex64)
        
        self.prespec(self.selected_weight, self.phantom_img)
        self.draw_graph(self.phantomView)
        
        
    def browse(self):
        self.file_path_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', " ", "(*.png *.jpg *.jpeg)")
        self.file_name, self.file_extension = path.splitext(self.file_path_name)
        
    def select_pd(self):
        self.selected_weight = self.weights[0]
        self.prespec(self.selected_weight, self.phantom_img)
        self.draw_graph(self.phantomView)
        
    def select_t1(self):
        self.selected_weight = self.weights[1]
        self.prespec(self.selected_weight, self.phantom_img)
        self.draw_graph(self.phantomView)
        
    def select_t2(self):
        self.selected_weight = self.weights[2]
        self.prespec(self.selected_weight, self.phantom_img)
        self.draw_graph(self.phantomView)
        
    def prespec(self, typo ,img):
        self.unique_img = np.unique(img)
        self.LUT = np.array([self.unique_img,
    				[ 250 ,170 ,85 ,0],
    			    [ 85  , 100  ,170 ,200]])
        edited_img = img.copy()
        if typo == 'PD':
            pass
        elif typo == 'T2':
            i=0
            for element in self.LUT[0]:
                edited_img[edited_img == element] =self.LUT[1,i]
                i =i+1
        elif typo == 'T1':
            i=0
            for element in self.LUT[0]:
                edited_img[edited_img == element] =self.LUT[2,i]
                i =i+1
        self.weighted_img = edited_img


    def draw_graph(self, phantomView):
        phantomView.clear()
        
        self.dimen_x = self.phantom_img.shape[0]
        self.dimen_y = self.phantom_img.shape[1]
        # print(self.dimen_x)
        
        # self.frame = np.random.rand(120, 100)
        # img = pg.ImageItem(self.weighted_img)
        
        # Interpret image data as row-major instead of col-major
        # pg.setConfigOptions(imageAxisOrder='row-major')
        # pg.mkQApp()
        
        # img = pg.ImageItem()

        # spec_plot2.addItem(img) # PlotWidget with ImageItem - WORKING BUT AS AN ITEM
        
        phantomView.setImage(self.weighted_img)        # ImgaeView
        # spec_plot2.scene().sigMouseClicked.connect(self.mouseClickEvent)

        #  # self.circle_item = CircleItem(pos=(0, 0), radius=0.5, brush=QBrush(QColor(100, 50, 50, 0)))
        # self.circle_item = pg.ScatterPlotItem(size=10, symbol='o', brush='w')
        
        # # Connect the mousePressEvent signal to the custom slot
        # self.phantomView.getView().mousePressEvent = self.handle_mouse_clicked
        
        # self.phantomView.sigMouseClicked.MouseClickEvent(self.mouseMovedEvent, double=False)

    def handle_mouse_clicked(self, event):
        # print('I\'M PRESSED')
        self.label_value.clear()
        
        
        # self.phantomView.getView().removeItem(self.circle_item)
        
        # # Create a CircleItem to display the mouse click position
        # self.circle_item = CircleItem(pos=(0, 0), radius=0.5, brush=QBrush(QColor(100, 50, 50, 0)))
        # self.phantomView.getView().addItem(self.circle_item)

        # Get the mouse click position in image coordinates
        pos = self.reconstructedView.getView().mapSceneToView(event.pos())
        mousePoint = pos
        self.x_i = round(mousePoint.x())
        self.y_i = round(mousePoint.y())
        if self.x_i > 0 and self.x_i < self.phantom_img.shape[1] and self.y_i > 0 and self.y_i < self.phantom_img.shape[0]:
            pd_value = self.phantom_img[self.y_i, self.x_i]
            itemindex = np.where(self.unique_img == pd_value)
            t1_value = self.LUT[1,itemindex[0][0]]
            t2_value = self.LUT[2,itemindex[0][0]]
            
            print(pd_value, t1_value, t2_value)
            self.label_value.setText("({}, {}) = PD{:0} T1:{:0} T2:{:0}".format(self.y_i, self.x_i, pd_value, t1_value, t2_value))
            return
        
        # # Update the CircleItem position and show it
        # self.circle_item.setPos(pos)
        # self.circle_item.setBrush(QBrush(QColor(255, 255, 255, 100)))
        
        # # Update the ScatterPlotItem position and show it
        # self.circle_item.setData(pos=[(pos.x(), pos.y())], brush=QtGui.QColor(255, 255, 255, 0))

        
        
    def start_kspace(self, kspaceView, reconstructedView):
        self.k_space = np.ones((self.num_of_rows, self.num_of_cols), dtype=np.complex64)
        kspaceView.clear()
        reconstructedView.clear()
        # Repeating timer, calls display_kspace over and over.
        self.kspacetimer = QtCore.QTimer()
        self.kspacetimer.setInterval(self.interval)
        self.kspacetimer.timeout.connect(lambda: self.display_kspace(kspaceView, reconstructedView, self.k_space))
        self.kspacetimer.start()

        # Single oneshot which stops the selection after 5 seconds
        QtCore.QTimer.singleShot((self.dimen_x + 1) * self.interval, self.stop_timer)

    def stop_timer(self):
        # Stop the random selection
        self.kspacetimer.stop()

    # the display function will be repeated Ny times according to the number of rows
    def display_kspace(self, kspaceView, reconstructedView, kspace):
        # kspaceView.clear()
        kspace_array = KSpace.build_kspace(KSpace(self.phantom_img), self.counter, kspace)
        # kspace_array[]
        # print(kspace_array)
        kspaceView.setImage(np.log(np.abs((kspace_array))))
        # print('kspace_array')
        self.reconstruct_image(reconstructedView, kspace_array)
        self.counter += 1
        if(self.counter >= self.dimen_x): self.counter = 0
        
    def reconstruct_image(self, reconstructedView, kspace):
        # print('reconstructed_image')
        reconstructedView.clear()
        image = np.abs((np.fft.ifft2(kspace)))
        reconstructedView.setImage(image)  # ImgaeView
        
        # Connect the mousePressEvent signal to the custom slot
        self.reconstructedView.getView().mousePressEvent = self.handle_mouse_clicked
       
        
    def hideHisto(self):
        for view in self.imageViews:
            view.ui.histogram.setFixedWidth(80)
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()

    def mouseClickEvent(self, mouseClickEvent):
        # Check if event is inside image, and convert from screen/pixels to image xy indicies
        # if self.plot.sceneBoundingRect().contains(pos):
        #     mousePoint = self.plot.getViewBox().mapSceneToView(pos)
        #     x_i = round(mousePoint.x())
        #     y_i = round(mousePoint.y())
        #     if x_i > 0 and x_i < self.frame.shape[0] and y_i > 0 and y_i < self.frame.shape[1]:
        #         self.label_value.setText("({}, {}) = {:0.2f}".format(x_i, y_i, self.frame[y_i, x_i,0]))
        #         return
        self.plot.scene().sigMouseMoved.connect(self.mouseMovedEvent)
        if self.x_i > 0 and self.x_i < self.frame.shape[0] and self.y_i > 0 and self.y_i < self.frame.shape[1]:
            self.label_value.clear()
            self.label_value.setText("({}, {}) = {:0.2f}".format(self.x_i, self.y_i, self.frame[self.y_i, self.x_i,0]))
        
    def mouseReleaseEvent(self, event):
        # ensure that the left button was pressed *and* released within the
        # geometry of the widget; if so, emit the signal;
        if (self.pressPos is not None and 
            event.button() == QtCore.Qt.LeftButton and 
            event.pos() in self.rect()):
                self.clicked.emit()
        self.pressPos = None

    def mouseMovedEvent(self, pos):
        # Check if event is inside image, and convert from screen/pixels to image xy indicies
        if self.phantomView.sceneBoundingRect().contains(pos):
            mousePoint = self.plot.getViewBox().mapSceneToView(pos)
            self.x_i = round(mousePoint.x())
            self.y_i = round(mousePoint.y())
            if self.x_i > 0 and self.x_i < self.phantom.shape[0] and self.y_i > 0 and self.y_i < self.phantom.shape[1]:
                self.label_value.setText("({}, {}) = {:0.2f}".format(self.x_i, self.y_i, self.phantom[self.y_i, self.x_i,0]))
                return
        self.label_value.clear()

    def phantom(self):
        pass

    def sequence(self):
        plt.clf()
        rf_amp = self.rfDial.value() / 45
        time_step=0.1
        start_time=50
        Gz_amp=0.4
        Gz_time=15
        Rf_time=Gz_time
        Gph_max_amp=.7
        Gph_time=7
        Gx_amp = 0.4
        Gx_time=24
        no_of_rows =5
        scaling_factor=2
        noise_scaling=5

        time = np.arange(0,60,time_step)
        zeros=np.full(time.shape,0,dtype=float)
        Rf_x_axis=int(Rf_time/time_step)
        fin_Gz_time_pos=int((Gz_time/time_step)+(start_time))
        fin_Gph_time=int(fin_Gz_time_pos+(Gph_time/time_step))
        fin_Gz_time_neg=int(fin_Gz_time_pos+(Gz_time/(2*time_step)))
        # str_Gx_time_neg=int(fin_Gph_time-(Gx_time/(2*time_step)))
        str_Gx_time_neg=fin_Gz_time_pos
        fin_Gx_time_neg=int(str_Gx_time_neg+(Gx_time/(2*time_step)))
        
        fin_Gx_time_pos=int(fin_Gx_time_neg+(Gx_time/time_step))
        fin_RO_time_pos=int(fin_Gx_time_pos-(Gx_time/(2*time_step))+(Rf_time/(2*time_step)))
        str_RO_time_pos=int(fin_RO_time_pos-(Rf_time/(time_step)))
        
        Gz_zeros=zeros.copy()
        
        Gz_zeros[start_time:fin_Gz_time_pos]=Gz_amp
        Gz_zeros[fin_Gz_time_pos:fin_Gz_time_neg]=-Gz_amp


        Gph_zeros=zeros.copy()
        for row in range(1,no_of_rows+1):
        	Gph_amp= ((Gph_max_amp/no_of_rows)*row)
        	Gph_zeros[fin_Gz_time_pos:fin_Gph_time]=Gph_amp
        	Gph_zeros_neg = - Gph_zeros
        	Gph_zeros = Gph_zeros +(2*scaling_factor)
        	Gph_zeros_neg = Gph_zeros_neg +(2*scaling_factor)
        	plt.plot(time,Gph_zeros)
        	plt.plot(time,Gph_zeros_neg)
        	Gph_zeros_neg=zeros.copy()
        	Gph_zeros=zeros.copy()

        x=np.linspace(int(-Rf_time/2),int(Rf_time/2),Rf_x_axis)
        y=np.sinc(x) * rf_amp

        Gx_zeros=zeros.copy()
        Gx_zeros[str_Gx_time_neg:fin_Gx_time_neg]=-Gx_amp
        Gx_zeros[fin_Gx_time_neg:fin_Gx_time_pos]=Gx_amp
        
        rf_zeros=zeros.copy()
        rf_zeros[start_time:Rf_x_axis+start_time]=y
        
        xx=x.shape[0]

        ran=np.random.rand(x.shape[0])/noise_scaling
        y_ran = y+ran
        RO_zeros=zeros.copy()
        RO_zeros[str_RO_time_pos:fin_RO_time_pos]=y_ran
        
        Gx_zeros = Gx_zeros + (1*scaling_factor)
        Gz_zeros = Gz_zeros + (3*scaling_factor)
        rf_zeros = rf_zeros + (4*scaling_factor)

        plt.plot(time,Gx_zeros)
        plt.plot(time,Gz_zeros)
        plt.plot(time,rf_zeros)
        plt.plot(time,RO_zeros)
        # ax = plt.gca()
        # ax.facecolor('#0F1414')
        plt.savefig('seq_img.png')
        image = "seq_img.png"
        pixmap = QtGui.QPixmap(image)
        self.sequenceLabel.setPixmap(pixmap)
        
        # plt.show()

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    # qdarktheme.setup_theme(
    #     custom_colors={
    #         "[light]": {
    #             "background": "#3f4042",
    #         }
    #     }
    # )
    mainwindow = MriMain()
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()



# import numpy as np
# from PyQt5 import QtWidgets
# import pyqtgraph as pg
# from PIL import Image
# from PyQt5.uic import loadUiType
# from PyQt5 import QtCore, QtGui
# from pyqtgraph import ImageView
# from PyQt5.QtGui import QBrush, QColor
# from os import path
# from kspace import KSpace
# from circle_item import CircleItem
# import cv2
# import qdarktheme
# import asyncio


# FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui_mri.ui"))
# class MriMain(QtWidgets.QMainWindow, FORM_CLASS):

#     def __init__(self):
#         # Interpret image data as row-major instead of col-major
#         pg.setConfigOptions(imageAxisOrder='row-major') 
#         pg.setConfigOption('background', (15,20, 20))
#         super(MriMain,self).__init__()

#         # Basic UI layout
#         self.setupUi(self)
#         self.statusbar = QtWidgets.QStatusBar(self)
#         self.setStatusBar(self.statusbar)
#         self.glw = pg.GraphicsLayoutWidget()
#         # self.setCentralWidget(self.glw)
        
#         self.phantom_img =cv2.imread("phantoms/brain_20p.jpg",0)
#         self.phantom_img = np.array(self.phantom_img)
#         self.num_of_rows = self.phantom_img.shape[0]
#         self.num_of_cols = self.phantom_img.shape[1]
#         self.k_space = np.ones((self.num_of_rows, self.num_of_cols), dtype=np.complex64)
        
#         # Async
#         self.loop = asyncio.get_event_loop()
#         self.task = self.loop.create_task(self.update_label())
        
        
#         self.reconstDefaultSize.clicked.connect(lambda: self.reconstructedView.autoRange())
#         self.phantomDefaultSize.clicked.connect(lambda: [self.phantomView.autoRange(), self.kspaceView.autoRange()])
#         self.actionPlay.triggered.connect(lambda: self.start_kspace(self.kspaceView, self.reconstructedView))
#         # self.markerCheckBox.stateChanged.connect(self.set_point)
        
#         # self.plot = self.phantomView.addPlot()
#         self.x_i = 0
#         self.y_i = 0
        
#         self.counter = 0
#         self.interval = 1000
#         # self.draw_graph(self.plot)
        
#         self.imageViews = [self.reconstructedView, self.phantomView, self.kspaceView]
#         self.hideHisto()
        
#         self.draw_graph(self.phantomView)
        

#         # Connect the mouseClicked signal to the custom slot
#         # self.reconstructedView.mouseClicked.connect(self.handle_mouse_clicked)

        
#     def draw_graph(self, phantomView):
#         phantomView.clear()
        
#         self.dimen_x = self.phantom_img.shape[0]
#         self.dimen_y = self.phantom_img.shape[1]
#         print(self.dimen_x)
        
#         # self.frame = np.random.rand(120, 100)
#         img = pg.ImageItem(self.phantom_img)
        
#         # Interpret image data as row-major instead of col-major
#         # pg.setConfigOptions(imageAxisOrder='row-major')
#         # pg.mkQApp()
        
#         # img = pg.ImageItem()

#         # spec_plot2.addItem(img) # PlotWidget with ImageItem - WORKING BUT AS AN ITEM
        
#         phantomView.setImage(self.phantom_img)        # ImgaeView
#         # spec_plot2.scene().sigMouseClicked.connect(self.mouseClickEvent)
        
       
#         # Connect the mousePressEvent signal to the custom slot
#         self.phantomView.getView().mousePressEvent = self.handle_mouse_clicked
       
#         #  # self.circle_item = CircleItem(pos=(0, 0), radius=0.5, brush=QBrush(QColor(100, 50, 50, 0)))
#         # self.circle_item = pg.ScatterPlotItem(size=10, symbol='o', brush='w')
        
#         # # Connect the mousePressEvent signal to the custom slot
#         # self.phantomView.getView().mousePressEvent = self.handle_mouse_clicked
        
#         # self.phantomView.sigMouseClicked.MouseClickEvent(self.mouseMovedEvent, double=False)

#     def handle_mouse_clicked(self, event):
#         print('I\'M PRESSED')
#         self.label_value.clear()
        
#         # self.phantomView.getView().removeItem(self.circle_item)
        
#         # # Create a CircleItem to display the mouse click position
#         # self.circle_item = CircleItem(pos=(0, 0), radius=0.5, brush=QBrush(QColor(100, 50, 50, 0)))
#         # self.phantomView.getView().addItem(self.circle_item)
        

#         # Get the mouse click position in image coordinates
#         pos = self.phantomView.getView().mapSceneToView(event.pos())
#         mousePoint = pos
#         self.x_i = round(mousePoint.x())
#         self.y_i = round(mousePoint.y())
#         if self.x_i > 0 and self.x_i < self.phantom_img.shape[0] and self.y_i > 0 and self.y_i < self.phantom_img.shape[1]:
#             self.label_value.setText("({}, {}) = {:0.2f}".format(self.x_i, self.y_i, self.phantom_img[self.y_i, self.x_i]))
#             return
        

#         # # Update the CircleItem position and show it
#         # self.circle_item.setPos(pos)
#         # self.circle_item.setBrush(QBrush(QColor(255, 255, 255, 100)))
        
#         # # Update the ScatterPlotItem position and show it
#         # self.circle_item.setData(pos=[(pos.x(), pos.y())], brush=QtGui.QColor(255, 255, 255, 0))

        
        
#     def start_kspace(self, kspaceView, reconstructedView):
#         self.k_space = np.ones((self.num_of_rows, self.num_of_cols), dtype=np.complex64)
#         kspaceView.clear()
#         reconstructedView.clear()
#         # Repeating timer, calls display_kspace over and over.
#         self.kspacetimer = QtCore.QTimer()
#         self.kspacetimer.setInterval(self.interval)
#         self.kspacetimer.timeout.connect(lambda: self.display_kspace(kspaceView, reconstructedView, self.k_space))
#         self.kspacetimer.start()

#         # Single oneshot which stops the selection after 5 seconds
#         QtCore.QTimer.singleShot((self.dimen_x + 1) * self.interval, self.stop_timer)

#     def stop_timer(self):
#         # Stop the random selection
#         self.kspacetimer.stop()

            
#     async def update_label(self):
#        for i in range(self.dimen_x):
#            value = await self.loop.run_in_executor(None, self.get_value)
#            self.label.config(text=f"Value: {value}")
#            await asyncio.sleep(1)
           
#     def get_value(self):
#         # This is an example function that returns a random number
#         self.kspaceView.setImage(np.log(np.abs((self.k_space))))
#         self.reconstruct_image(self.reconstructedView, self.k_space)
    
#     def display_kspace(self, kspaceView, reconstructedView, kspace):
        
#         # kspaceView.clear()
#         self.k_space = KSpace.build_kspace(KSpace(), self.counter, kspace)
#         # kspace_array[]
#         # print(kspace_array)
#         kspaceView.setImage(np.log(np.abs((self.k_space))))
#         print('kspace_array')
#         self.reconstruct_image(reconstructedView, self.k_space)
#         self.counter += 1
#         if(self.counter >= self.dimen_x): self.counter = 0
        
#         return 
        
        
#     def reconstruct_image(self, reconstructedView, kspace):
#         print('reconstructed_image')
#         reconstructedView.clear()
#         image = np.abs((np.fft.ifft2(kspace)))
#         reconstructedView.setImage(image)  # ImgaeView
        
#     def hideHisto(self):
#         for view in self.imageViews:
#             view.ui.histogram.setFixedWidth(80)
#             view.ui.roiBtn.hide()
#             view.ui.menuBtn.hide()
        
#     # def handle_mouse_clicked(self, event):
#     #     # Get the mouse click position in image coordinates
#     #     pos = self.getImageItem().mapFromScene(event.pos())

#     #     # Update the CircleItem position and show it
#     #     self.circle_item.setPos(pos)
#     #     self.circle_item.setBrush(QBrush(QColor(255, 255, 255, 100)))
        
#     def mouseClickEvent(self, mouseClickEvent):
#         # Check if event is inside image, and convert from screen/pixels to image xy indicies
#         # if self.plot.sceneBoundingRect().contains(pos):
#         #     mousePoint = self.plot.getViewBox().mapSceneToView(pos)
#         #     x_i = round(mousePoint.x())
#         #     y_i = round(mousePoint.y())
#         #     if x_i > 0 and x_i < self.frame.shape[0] and y_i > 0 and y_i < self.frame.shape[1]:
#         #         self.label_value.setText("({}, {}) = {:0.2f}".format(x_i, y_i, self.frame[y_i, x_i,0]))
#         #         return
#         self.plot.scene().sigMouseMoved.connect(self.mouseMovedEvent)
#         if self.x_i > 0 and self.x_i < self.frame.shape[0] and self.y_i > 0 and self.y_i < self.frame.shape[1]:
#             self.label_value.clear()
#             self.label_value.setText("({}, {}) = {:0.2f}".format(self.x_i, self.y_i, self.frame[self.y_i, self.x_i,0]))
        
#     def mouseReleaseEvent(self, event):
#         # ensure that the left button was pressed *and* released within the
#         # geometry of the widget; if so, emit the signal;
#         if (self.pressPos is not None and 
#             event.button() == QtCore.Qt.LeftButton and 
#             event.pos() in self.rect()):
#                 self.clicked.emit()
#         self.pressPos = None

#     def mouseMovedEvent(self, pos):
#         # Check if event is inside image, and convert from screen/pixels to image xy indicies
#         if self.phantomView.sceneBoundingRect().contains(pos):
#             mousePoint = self.plot.getViewBox().mapSceneToView(pos)
#             self.x_i = round(mousePoint.x())
#             self.y_i = round(mousePoint.y())
#             if self.x_i > 0 and self.x_i < self.phantom.shape[0] and self.y_i > 0 and self.y_i < self.phantom.shape[1]:
#                 self.label_value.setText("({}, {}) = {:0.2f}".format(self.x_i, self.y_i, self.phantom[self.y_i, self.x_i,0]))
#                 return
#         self.label_value.clear()


# def main():
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     qdarktheme.setup_theme("dark")
#     # qdarktheme.setup_theme(
#     #     custom_colors={
#     #         "[light]": {
#     #             "background": "#3f4042",
#     #         }
#     #     }
#     # )
#     mainwindow = MriMain()
#     mainwindow.show()
#     sys.exit(app.exec_())

# if __name__ == '__main__':
#     main()
