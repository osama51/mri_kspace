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
from kspace import Parameters
from kspace import Prep_Pulses
from kspace import ACQ_Seq
from seq_plot import Plotting
from circle_item import CircleItem
import sequence
import phantom
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Stop: 
    should_stop = False
    counter = 0
    restart = False
    
    
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
    shared_variables['kspace'] = KSpace.build_kspace(KSpace(shared_variables['phantom_img']), 
                                                     Stop.counter, shared_variables['kspace'], 
                                                     shared_variables['selected_prep'], 
                                                     shared_variables['selected_seq'])
    
    final_k = np.log(np.abs((shared_variables['kspace'])))
    shared_variables['kspaceViews'][shared_variables['selected_port']].setImage(final_k)
    shared_variables['kspaceViews'][shared_variables['selected_port']].setLevels(final_k.min(), final_k.max())
    
    reconstruct_image(shared_variables['phantomViews'][shared_variables['selected_port']], shared_variables['kspace'])
    Stop.counter += 1
    print("counter", Stop.counter)
    if(Stop.counter >= shared_variables['dimen_x']): 
        Stop.counter = 0
        Stop.should_stop = True
        init_kspace()
        shared_variables['actionReconstruct'].setEnabled(True)
        shared_variables['actionStop'].setEnabled(False)
        shared_variables['actionPause'].setEnabled(False)
        print("I'm done, do you witness the greatness?\n")
        
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


FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui_mri.ui"))
class MriMain(QtWidgets.QMainWindow, FORM_CLASS):

    def __init__(self, instance_of_class_a):
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
        
        self.plotting = instance_of_class_a
        
        self.file_path_name = ""
        
        self.weights = ["PD", "T1", "T2"]
        self.selected_weight = self.weights[0]
        self.pdRadioButton.toggled.connect(self.select_pd)
        self.t1RadioButton.toggled.connect(self.select_t1)
        self.t2RadioButton.toggled.connect(self.select_t2)
        
        self.selected_size = 0
        self.sizeComboBox.currentIndexChanged.connect(self.select_size)
        
        self.reconstDefaultSize.clicked.connect(self.autoRange_reconstructedViews)
        self.phantomDefaultSize.clicked.connect(lambda: [self.phantomView.autoRange(), 
                                                         self.kspaceView.autoRange()])
        self.actionReconstruct.triggered.connect(lambda: self.start_kspace(self.kspaceViews[self.selected_port], 
                                                                           self.phantomViews[self.selected_port]))
        self.actionStop.triggered.connect(self.stop_timer)
        self.actionPause.triggered.connect(self.pause_timer)
        self.actionOpen.triggered.connect(self.browse)
        # self.markerCheckBox.stateChanged.connect(self.set_point)
        
        self.TR_slider.valueChanged.connect(self.assign_TR_TE)
        self.TE_slider.valueChanged.connect(self.assign_TR_TE)
        self.RF_slider.valueChanged.connect(self.assign_TR_TE)
        self.TI_slider.valueChanged.connect(self.assign_TR_TE)
        self.duration_slider.valueChanged.connect(self.assign_TR_TE)
        self.angle_slider.valueChanged.connect(self.assign_TR_TE)
        self.width_slider.valueChanged.connect(self.assign_TR_TE)
        # self.RF_slider.valueChanged.connect(self.sequence)
        
        self.RF_slider.valueChanged.connect(self.seq_plot_new)
        self.TE_slider.valueChanged.connect(self.seq_plot_new)
        self.TI_slider.valueChanged.connect(lambda: self.plotting.prep_inv(self.TI_slider.value(), self.seq_graphicsView))
        self.duration_slider.valueChanged.connect(lambda: self.plotting.prep_T2(self.duration_slider.value(), self.seq_graphicsView))
        self.angle_slider.valueChanged.connect(lambda: self.plotting.prep_inv(self.angle_slider.value(), 20, 10, self.seq_graphicsView))


        
        self.prep_comboBox.currentIndexChanged.connect(self.prep_index_changed)
        self.seq_comboBox.currentIndexChanged.connect(self.seq_index_changed)
        self.port_comboBox.currentIndexChanged.connect(self.port_index_changed)
        
        self.actionPause.setEnabled(False)
        self.actionStop.setEnabled(False)
        print("for some reason I restarted!")
        # self.plot = self.phantomView.addPlot()
        self.x_i = 0
        self.y_i = 0
        
        self.counter = 0
        self.interval = 2000
        self.should_stop = False
        self.selected_prep = Prep_Pulses.NONE
        self.selected_seq = ACQ_Seq.GRE
        self.selected_port = 0
        # self.draw_graph(self.plot)
        
        self.imageViews = [self.reconstructedView, self.reconstructedView_2, self.phantomView, self.kspaceView, self.kspaceView_2]
        
        self.kspaceViews = [self.kspaceView, self.kspaceView_2]
        self.phantomViews = [self.reconstructedView, self.reconstructedView_2]
        self.hideHisto()
        
        self.assign_TR_TE()
        self.hide_sliders()
        self.select_size(self.selected_size)
        self.sequence()

        # Connect the mouseClicked signal to the custom slot
        # self.reconstructedView.mouseClicked.connect(self.handle_mouse_clicked)
    
    def assign_TR_TE(self):
        Parameters.TR = self.TR_slider.value()
        Parameters.TE = self.TE_slider.value()
        Parameters.RF = self.RF_slider.value()
        Parameters.TI = self.TI_slider.value()
        Parameters.duration = self.duration_slider.value()
        Parameters.Angle = self.angle_slider.value() * 45
        Parameters.Width = self.width_slider.value()
        
        self.TR_label.setText(str(self.TR_slider.value()))
        self.TE_label.setText(str(self.TE_slider.value()))
        self.RF_label.setText(str(self.RF_slider.value()))
        self.TI_label.setText(str(self.TI_slider.value()))
        self.duration_label.setText(str(self.duration_slider.value()))
        self.angle_label.setText(str(self.angle_slider.value() * 45))
        self.width_label.setText(str(self.width_slider.value()))
        
        self.seq_plot_new()
        
    def hide_sliders(self):
        self.prep_groupBox.hide()
        
        self.TI_slider.hide()
        self.TI_label.hide()
        self.label_TI.hide()
        
        self.duration_slider.hide()
        self.duration_label.hide()
        self.label_duration.hide()
        
        self.angle_slider.hide()
        self.angle_label.hide()
        self.label_angle.hide()
        
        self.width_slider.hide()
        self.width_label.hide()
        self.label_width.hide()
        
        
    def show_IR(self):
        self.hide_sliders()
        self.prep_groupBox.show()
        self.TI_slider.show()
        self.TI_label.show()
        self.label_TI.show()
    
    def show_T2(self):
        self.hide_sliders()
        self.prep_groupBox.show()
        self.duration_slider.show()
        self.duration_label.show()
        self.label_duration.show()
        
    def show_Tagging(self):
        self.hide_sliders()
        self.prep_groupBox.show()
        self.angle_slider.show()
        self.angle_label.show()
        self.label_angle.show()
        
        # self.width_slider.show()
        # self.width_label.show()
        # self.label_width.show()
        
    def prep_index_changed(self, index):
        print("yes yes I hear you stop shouting ffs")
        if(index==0):
            self.hide_sliders()
            self.selected_prep = Prep_Pulses.NONE
            self.seq_plot_new()
        elif(index==1):
            self.show_IR()
            self.selected_prep = Prep_Pulses.IR
            self.plotting.prep_inv(self.TI_slider.value(), self.seq_graphicsView)
        elif(index==2):
            self.show_T2()
            self.selected_prep = Prep_Pulses.T2
            self.plotting.prep_T2(self.duration_slider.value(), self.seq_graphicsView)
        elif(index==3):
            self.show_Tagging()
            self.selected_prep = Prep_Pulses.TAGGING
            self.plotting.tagging_draw(self.angle_slider.value() * 45, 20, 10,self.seq_graphicsView)
            
    def seq_index_changed(self, index):
        if(index==0):
            self.selected_seq = ACQ_Seq.GRE
            pass
        elif(index==1): # Not Implemented 
            self.selected_seq = ACQ_Seq.SPOILED_GRE
            pass
        elif(index==2): # Not Implemented 
            self.selected_seq = ACQ_Seq.BALANCED
            pass
        elif(index==3): 
            self.selected_seq = ACQ_Seq.SE
            pass
        elif(index==4): # Not Implemented 
            self.selected_seq = ACQ_Seq.TSE
            pass
        
    def port_index_changed(self, index):
        if(index==0):
            self.selected_port = 0
        elif(index==1):
            self.selected_port = 1

        
    def autoRange_reconstructedViews(self):
        for i in self.phantomViews:
            i.autoRange()
        for i in self.kspaceViews:
            i.autoRange()
            
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
        print("I'm so unique", self.unique_img)
        self.LUT = np.array([self.unique_img,
    				[ 0 ,102 ,84 ,1],
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
            'should_stop': self.should_stop,
            'selected_prep': self.selected_prep,
            'selected_seq': self.selected_seq,
            'selected_port': self.selected_port,
            'kspaceViews': self.kspaceViews,
            'phantomViews': self.phantomViews
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
        reconstructedView.setLevels(image.min(), image.max())
        
        # Connect the mousePressEvent signal to the custom slot
        self.reconstructedView.getView().mousePressEvent = self.handle_mouse_clicked
       
        
    def hideHisto(self):
        self.seq_graphicsView.hideAxis('bottom')
        for view in self.imageViews:
            view.setLevels(0, 255)
            view.ui.histogram.setFixedWidth(70)
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            
        for view in self.kspaceViews:
            view.setLevels(0, 10)

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
        TR = self.TR_slider.value()
        TE = self.TE_slider.value()
        rf_amp = self.RF_slider.value() / 45
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
        # self.sequenceLabel.setPixmap(pixmap)
        
        # plt.show()
        
    def seq_plot_new(self):
        self.plotting.Rf_amp_inv = self.RF_slider.value() / 45
        self.plotting.TE = self.TE_slider.value() 
      
        # print("Rf_amp_inv", self.plotting.Rf_amp_inv)
        self.seq_graphicsView.clear()
        self.plotting.draw_Gy(self.seq_graphicsView)
        self.plotting.draw_the_rest(self.seq_graphicsView)
        
        if(self.selected_prep==Prep_Pulses.NONE):
            pass
        elif(self.selected_prep==Prep_Pulses.IR):
            self.plotting.prep_inv(self.TI_slider.value(), self.seq_graphicsView)
        elif(self.selected_prep==Prep_Pulses.T2):
            self.plotting.prep_T2(self.duration_slider.value(), self.seq_graphicsView)
        elif(self.selected_prep==Prep_Pulses.TAGGING):
            self.plotting.tagging_draw(self.angle_slider.value() * 45, 20, 10,self.seq_graphicsView)
            

            

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
    plotting = Plotting()
    mainwindow = MriMain(plotting)
    mainwindow.show()

    # Run the PyQt event loop
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()
