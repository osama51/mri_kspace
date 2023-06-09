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
import threading
import asyncio
from kspace import KSpace
from circle_item import CircleItem
import sequence
import phantom
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Stop: 
    should_stop = False
    counter = 0
    restart = False
    
#multithreading functions
class ProcessRunnable(QRunnable):
    def __init__(self, target, args):
        QRunnable.__init__(self)
        self.target = target
        self.args = args

    def run(self):
        self.target(*self.args)

    def start(self):
        QThreadPool.globalInstance().start(self)


# the display function will be repeated Ny times according to the number of rows
def display_kspace(shared_variables, reconstruct_image, phantom_img, init_kspace):
    shared_variables['kspace'] = KSpace.build_kspace(KSpace(shared_variables['phantom_img']), Stop.counter, shared_variables['kspace'])
    shared_variables['kspace_view'].setImage(np.log(np.abs((shared_variables['kspace']))))
    
    reconstruct_image(shared_variables['reconstructed_view'], shared_variables['kspace'])
    Stop.counter += 1
    print("counter", Stop.counter)
    if(Stop.counter >= shared_variables['dimen_x']): 
        Stop.counter = 0
        Stop.should_stop = True
        init_kspace()
        # self.stop_timer()
        # self.stop_thread()
        shared_variables['actionReconstruct'].setEnabled(True)
        shared_variables['actionStop'].setEnabled(False)
        shared_variables['actionPause'].setEnabled(False)
        print("I'm done, do you witness the greatness?\nYES! WHAT A BRAIN!")
        
    print("should stop: ", Stop.should_stop)
    while not Stop.should_stop:
        display_kspace(shared_variables, reconstruct_image, phantom_img, init_kspace)
        
    if(Stop.restart):
        print("yes I got restarted")
        Stop.counter = 0
        Stop.should_stop = True
        init_kspace()
        Stop.restart = False
    # if(Stop.should_stop): Stop.should_stop = False


FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui_edit.ui"))
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
        self.actionReconstruct.triggered.connect(lambda: self.start_kspace(self.kspaceView, self.reconstructedView))
        self.actionStop.triggered.connect(self.stop_timer)
        self.actionPause.triggered.connect(self.pause_timer)
        self.actionOpen.triggered.connect(self.browse)
        # self.markerCheckBox.stateChanged.connect(self.set_point)
        self.T2_Parameters.hide()
        self.Tagging_Parameters.hide()
        self.INV_Parameters.hide()
        self.SSFP_Parameters.hide()
        self.GRE_Parameters.hide()
        self.SE_Parameters.hide()
        self.TSE_Parameters.hide()
        self.verticalSlider_4.valueChanged.connect(self.sequence)
        self.INV_Prep.toggled.connect(self.hidebox1)
        self.T2_Prep.toggled.connect(self.hidebox2  )
        self.Tagging_Prep.toggled.connect(self.hidebox3)
        self.none.toggled.connect(self.hidebox4)
        self.actionPause.setEnabled(False)
        self.actionStop.setEnabled(False)
        print("for some reason I restarted!")
        # self.plot = self.phantomView.addPlot()
        self.x_i = 0
        self.y_i = 0
        
        self.counter = 0
        self.interval = 2000
        self.should_stop = False
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
    				[ 254 ,102 ,84 ,1],
    			    [ 83  , 100  ,169 ,200]])
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
            # print(edited_img)
        self.weighted_img = edited_img


    def draw_graph(self, phantomView):
        phantomView.clear()
        
        self.dimen_x = self.phantom_img.shape[0]
        self.dimen_y = self.phantom_img.shape[1]
        
        phantomView.setImage(self.weighted_img)        # ImgaeView
    

    def handle_mouse_clicked(self, event):
        self.label_value.clear()

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
    
    def init_kspace(self):
        print("fshhhhhh. kspace flushed!")
        self.k_space = np.ones((self.num_of_rows, self.num_of_cols), dtype=np.complex64)
        return self.k_space
        
    def start_kspace(self, kspaceView, reconstructedView):
        self.should_stop = False
        Stop.should_stop = False
        self.actionReconstruct.setEnabled(False)
        self.actionStop.setEnabled(True)
        self.actionPause.setEnabled(True)
        kspaceView.clear()
        reconstructedView.clear()
        # # Repeating timer, calls display_kspace over and over.
        # self.kspacetimer = QtCore.QTimer()
        # self.kspacetimer.setInterval(self.interval)
        # self.kspacetimer.timeout.connect(lambda: self.display_kspace(kspaceView, reconstructedView, self.k_space))
        # self.kspacetimer.start()
        
        # kspace_view = kspaceView
        # reconstructed_view = reconstructedView
        # kspace = self.k_space
        # phantom_img = self.phantom_img
        # counter = self.counter
        # dimen_x = self.dimen_x
        # actionReconstruct = self.actionReconstruct
        # actionStop = self.actionStop
        # actionPause = self.actionPause
            

        self.shared_variables = {
            'kspace_view': kspaceView,
            'reconstructed_view': reconstructedView,
            'kspace': self.k_space,
            'phantom_img': self.phantom_img,
            'counter': Stop.counter,
            'dimen_x': self.dimen_x,
            'actionReconstruct': self.actionReconstruct,
            'actionStop': self.actionStop,
            'actionPause': self.actionPause,
            'should_stop': self.should_stop
            }

        
        # Create and start the background task
        task = ProcessRunnable(target=display_kspace, args=(self.shared_variables, self.reconstruct_image, self.phantom_img, self.init_kspace))
        task.start()


    def stop_timer(self):
        # Stop the random selection
        Stop.counter = 0
        Stop.restart = True
        self.init_kspace()
        # self.kspacetimer.stop()
        self.stop_background_process()

        self.actionReconstruct.setEnabled(True)
        self.actionStop.setEnabled(False)
        print("I stopped")
        
    def pause_timer(self):
        # self.kspacetimer.stop()
        self.stop_background_process()
        self.actionPause.setEnabled(False)
        self.actionReconstruct.setEnabled(True)
        
    def stop_background_process(self):
        Stop.should_stop = True

        
    def reconstruct_image(self, reconstructedView, kspace):
        # print('reconstructed_image')
        reconstructedView.clear()
        image = np.abs((np.fft.ifft2(kspace)))
        reconstructedView.setImage(image)  # ImgaeView
        
        # Connect the mousePressEvent signal to the custom slot
        self.reconstructedView.getView().mousePressEvent = self.handle_mouse_clicked
       
    def hidebox1 (self):
        self.INV_Parameters.show()
        self.T2_Parameters.hide()
        self.Tagging_Parameters.hide()
        

    def hidebox2 (self):
        self.INV_Parameters.hide()
        self.T2_Parameters.show()
        self.Tagging_Parameters.hide()

    def hidebox3 (self):
        self.INV_Parameters.hide()
        self.T2_Parameters.hide()
        self.Tagging_Parameters.show()
    
    def hidebox4(self):
        self.INV_Parameters.hide()
        self.T2_Parameters.hide()
        self.Tagging_Parameters.hide()
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
        rf_amp = self.verticalSlider_4.value() / 45
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
            Gph_zeros[fin_Gz_time_pos:fin_Gph_time]= Gph_amp
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

    mainwindow = MriMain()
    mainwindow.show()

    # Run the PyQt event loop
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()
