# import sys
# import wave
# import threading
# import numpy as np
# from os import path
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from PyQt5 import QtWidgets
# import matplotlib.pyplot as plt
# import time
# from PyQt5.uic import loadUiType
# from PyQt5 import QtCore, QtGui
# from scipy.fft import fft, fftfreq, rfft, rfftfreq
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.uic import loadUiType
# from PyQt5.QtCore import QTimer
# from os import path
# import pyqtgraph
# from scipy import signal
# import pyqtgraph as pg
# from time import sleep

# import scipy
# # import simpleaudio as sa
# import sounddevice as sd
# import logging
# # import threading
# import _thread

# # import shutil

# # import soundfile as sf
# # from PyQt5 import QtCore, QtGui, QtWidgets
# # from pyqtgraph import PlotWidget
# # import pyqtgraph as pg

# # from pop import popWindow
# # from scipy.io import wavfile
# # import numpy as np
# # import sys
# # import os
# # from scipy.fftpack import fft




# FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui.ui"))
# class MainApp(QMainWindow, FORM_CLASS):
#     def __init__(self , parent=None):
#         pyqtgraph.setConfigOption('background', (0,0,0))
#         super(MainApp,self).__init__(parent)
#         QMainWindow.__init__(self)
#         self.setupUi(self)
#         self.iterator = 0
#         self.timor= 0.0 
#         self.pen=pyqtgraph.mkPen(color='c') 
#         self.timer = QTimer(self)
#         self.grMain.plotItem.showGrid(True, True, 0.7)
#         self.grUpdatedMain.plotItem.showGrid(True, True, 0.7)
#         # self.grSpec.plotItem.showGrid(True, True, 0.7)
#         # self.handle_buttons()
#         self.verticalSlider_gain.valueChanged.connect(self.gain)
#         self.verticalSlider_60.valueChanged.connect(
#             lambda:self.equalizer(0,60,self.verticalSlider_60.value()))
#         self.verticalSlider_170.valueChanged.connect(
#             lambda:self.equalizer(60,250,self.verticalSlider_170.value()))
#         self.verticalSlider_310.valueChanged.connect(
#             lambda:self.equalizer(250,400,self.verticalSlider_310.value()))
#         self.verticalSlider_600.valueChanged.connect(
#             lambda:self.equalizer(400,700,self.verticalSlider_600.value()))
#         self.verticalSlider_1k.valueChanged.connect(
#             lambda:self.equalizer(700,2000,self.verticalSlider_1k.value()))
#         self.verticalSlider_3k.valueChanged.connect(
#             lambda:self.equalizer(2000,4000,self.verticalSlider_3k.value()))
#         self.verticalSlider_6k.valueChanged.connect(
#             lambda:self.equalizer(4000,6000,self.verticalSlider_6k.value()))
#         self.verticalSlider_10k.valueChanged.connect(
#             lambda:self.equalizer(6000,10000,self.verticalSlider_10k.value()))
#         self.verticalSlider_16k.valueChanged.connect(
#             lambda:self.equalizer(10000,16000,self.verticalSlider_16k.value()))
        
#         self.actionOpen.triggered.connect(self.browse)
#         self.actionPlay.triggered.connect(self.toggle)
#         self.actionStop.triggered.connect(self.stop)
        
#         # ###############xylofone buttons###############
#         self.xylophoneC2.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/C2_xyl'))
#         self.xylophoneD2.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/D2_xyl'))
#         self.xylophoneE2.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/E2_xyl'))
#         self.xylophoneF2.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/F2_xyl'))
#         self.xylophoneG2.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/G2_xyl'))
#         self.xylophoneA2.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/A2_xyl'))
#         self.xylophoneB2.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/B2_xyl'))
#         self.xylophoneAsharp2.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/Asharp2_xyl'))
#         self.xylophoneC3.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/C3_xyl'))
#         self.xylophoneD3.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/D3_xyl'))
#         self.xylophoneE3.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/E3_xyl'))
#         self.xylophoneF3.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/F3_xyl'))
#         self.xylophoneG3.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/G3_xyl'))
#         self.xylophoneA3.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/A3_xyl'))
#         self.xylophoneB3.clicked.connect(lambda:self.read_csv_and_play_audio('xylophone_to_csv/B3_xyl'))
#         ###############stringsteel buttons###############
#         self.stsC5.clicked.connect(lambda:self.read_csv_and_play_audio('stringsteel_to_csv/c5_stst'))
#         self.stsD5.clicked.connect(lambda:self.read_csv_and_play_audio('stringsteel_to_csv/d5_stst'))
#         self.stsE5.clicked.connect(lambda:self.read_csv_and_play_audio('stringsteel_to_csv/e5_stst'))
#         self.stsF5.clicked.connect(lambda:self.read_csv_and_play_audio('stringsteel_to_csv/f5_stst'))
#         self.stsG5.clicked.connect(lambda:self.read_csv_and_play_audio('stringsteel_to_csv/g5_stst'))
#         self.stsA5.clicked.connect(lambda:self.read_csv_and_play_audio('stringsteel_to_csv/a5_stst'))
#         self.stsA5.clicked.connect(lambda:self.read_csv_and_play_audio('stringsteel_to_csv/a5_stst'))
#         ###############piano buttons###############
#         self.pianoC4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/C4_piano'))
#         self.pianoCsharp4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/Csharp4_piano'))
#         self.pianoD4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/D4_piano'))
#         self.pianoDsahrp4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/Dsharp4_piano'))
#         self.pianoE4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/E4_piano'))
#         self.pianoF4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/F4_piano'))
#         self.pianoFsharp4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/Fsharp4_piano'))
#         self.pianoG4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/G4_piano'))
#         self.pianoGsharp4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/Gsharp4_piano'))
#         self.pianoA4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/A4_piano'))
#         self.pianoAsharp4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/Asharp4_piano'))
#         self.pianoB4.clicked.connect(lambda:self.read_csv_and_play_audio('piano_to_csv/B4_piano'))
        
#     def handle_slider(self):
#         self.label_60.setText(str(self.verticalSlider_60.value()))
#         self.label_170.setText(str(self.verticalSlider_170.value()))
#         self.label_310.setText(str(self.verticalSlider_310.value()))
#         self.label_600.setText(str(self.verticalSlider_600.value()))
#         self.label_1k.setText(str(self.verticalSlider_1k.value()))
#         self.label_3k.setText(str(self.verticalSlider_3k.value()))
#         self.label_6k.setText(str(self.verticalSlider_6k.value()))
#         self.label_10k.setText(str(self.verticalSlider_10k.value()))
#         self.label_16k.setText(str(self.verticalSlider_16k.value()))

#     def reset_sliders(self):
#         self.verticalSlider_60.setValue(0)
#         self.verticalSlider_170.setValue(0)
#         self.verticalSlider_310.setValue(0)
#         self.verticalSlider_600.setValue(0)
#         self.verticalSlider_1k.setValue(0)
#         self.verticalSlider_3k.setValue(0)
#         self.verticalSlider_6k.setValue(0)
#         self.verticalSlider_10k.setValue(0)
#         self.verticalSlider_16k.setValue(0)
            
        
#     def browse(self):
        
#         self.file_path_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', " ", "(*.txt *.csv *.xls *.wav)")
#         self.file_name, self.file_extension = os.path.splitext(self.file_path_name)
#         self.read_data()
#         print("I read the data!")

#     def read_data(self):
#         self.reset_sliders()
#         self.iterator = 0
#         spf = wave.open(self.file_path_name, "r")
#         # print(spf)
#         # print(type(spf))
#         # Extract Raw Audio from Wav File
#         self.original_signal = spf.readframes(-1)
#         # print(type(self.original_signal))
#         self.original_signal = np.frombuffer(self.original_signal, "int16")
#         self.signal=self.original_signal
#         # print(type(self.original_signal))
#         # print(self.original_signal[5000], 'heeeeeeeeeere')
#         self.samplerate = spf.getframerate()
#         self.one_frame = int(self.samplerate/10) #17
#         self.Time = np.linspace(0, len(self.original_signal) / self.samplerate, num=len(self.original_signal))
#         self.original_spectrum = rfft(self.original_signal)[:len(self.original_signal)]
#         # self.original_spectrum = list(self.original_spectrum)
#         self.spectrum = self.original_spectrum
#         # print(self.spectrum[5000], type(self.spectrum[0]))
#         # print(scipy.fft.ifft(self.spectrum)[5000], type(self.spectrum[0]))
#         self.freq = rfftfreq(len(self.original_spectrum),1/self.samplerate)[:len(self.original_signal)]

#     def read_csv_and_play_audio(self,note_name):
#         path=note_name+'.csv'
#         data = pd.read_csv(path)
#         real_spect=data.iloc[:, 0]
#         imag=data.iloc[:,1]
#         spectrum=[]
#         for i in range(len(real_spect)):
#             spectrum.append(complex(real_spect[i],imag[i]))
#         new_track= np.fft.irfft(spectrum)
#         samplerate = 48000                
#         # Ensure that highest value is in 16-bit range
#         audio_samples1 = new_track.real * (2**14 - 1) / np.max(new_track.real)
#         # Convert to 16-bit data
#         audio_samples1 = audio_samples1.astype(np.int16)
        
#         sd.play(audio_samples1, samplerate)
           
#     def equalizer(self,min_freq,max_freq,slider_value):
#         # sleep(0.5)
#         # self.handle_slider()
#         freq_list=list(self.freq)
#         first_index = 0
#         second_index = 0
#         db=slider_value
#         # print(db, 'dB')
#         # print(self.spectrum[int((first_index+second_index)/2)], 'not original')
#         for frequency in freq_list :
#             if first_index == 0 and frequency>min_freq:
#                 first_index = freq_list.index(frequency)

#             if second_index == 0 and frequency>max_freq:
#                 second_index = freq_list.index(frequency)
#         # print(self.original_spectrum[int((first_index+second_index)/2)], 'original222')
#         # print(self.spectrum[int((first_index+second_index)/2)], 'bef')
#         # print(first_index, second_index, 'ssssssss')
#         self.spectrum[first_index:second_index] = [x * pow(10, (db/10)) for x in self.original_spectrum[first_index:second_index]]

#         self.filtered_data = np.fft.irfft(self.spectrum)
#         # print(len(self.signal), 'before filter')
#         # print(self.filtered_data[5000])
#         # print(self.spectrum[5000])
#         # self.filtered_data = self.filtered_data.real * (2**14 - 1) / np.max(np.abs(self.filtered_data.real))
#         audio_samples2 = self.filtered_data.astype(np.int16)
#         self.signal= audio_samples2
#         # print(len(self.signal), 'after filter')
#         # print(len(self.original_signal), 'after filter')
#         # self.play_sound()
#         # index = int((time.time()-self.first_time)*self.samplerate)
#         # sd.play(self.signal.real[self.iterator:], self.samplerate)
#         # self.draw_spectrogram()
#         # sf.write('bass.wav', self.signal, self.samplerate)
#         self.updated_graphs()
        
#     # def fttttt(self):
#     #     pen=pyqtgraph.mkPen(color='r')
#     #     self.grSpec.plot(self.freq,abs(self.spectrum),pen=pen,clear=True)
        
#     def gain(self):
#         self.label_master_gain.setText(str(self.verticalSlider_gain.value()))
#         gain_ratio = float(self.verticalSlider_gain.value()/100)
#         print(gain_ratio, 'ratio')
#         # print('before',self.signal[431882])
#         self.signal = self.original_signal * gain_ratio

#         audio_samples2 = self.signal.astype(np.int16)
#         self.signal = audio_samples2
#         # sf.write('filename.wav', audio_samples2, self.samplerate)
        
#         self.updated_graphs()
#         # index = int((time.time()-self.first_time)*self.samplerate)
#         # sd.play(self.signal.real[self.iterator:], self.samplerate)
    
#     def realtime_toggle(self):
#         if self.actionRealtime.isChecked():
#             self.grUpdatedMain.plotItem.setXRange(self.Time[self.iterator], self.Time[self.iterator+self.one_frame])
#             self.grMain.plotItem.setXRange(self.Time[self.iterator], self.Time[self.iterator+self.one_frame])
#         else:
#             self.grUpdatedMain.plotItem.setXRange(self.Time[0], self.Time[len(self.Time)-1])
#             self.grMain.plotItem.setXRange(self.Time[0], self.Time[len(self.Time)-1])
            
            
#     def play_sound(self):
#         # self.gain()
#         self.first_time = time.time()
#         # sd.play(self.signal[self.iterator:], self.samplerate)
#         sd.play(self.signal.real[self.iterator:], self.samplerate)

#     def update(self):
#         self.realtime_toggle()
#         # self.grMain.plotItem.setXRange(self.Time[self.iterator], self.Time[self.iterator+self.one_frame])
#         self.iterator += self.one_frame
        
#         if self.iterator >= len(self.signal)-self.one_frame: 
#             self.iterator = 0
#             self.actionPlay.setChecked(False)
#         if self.actionPlay.isChecked():
#             self.timer.singleShot(100, self.update) #59
#         else:
#             self.first = time.time()
#             self.timer.stop()
#             sd.stop()
            
#     def updated_graphs(self):
#         self.draw_spectrogram(self.signal, self.grUpdatedSpec)
#         self.grUpdatedMain.plotItem.setYRange(min(self.original_signal.real)*2, max(self.original_signal.real)*2)
#         self.grUpdatedMain.clear()
#         self.grUpdatedMain.plot(self.Time, self.signal.real, pen = self.pen)
#         self.realtime_toggle()
        
#     def play(self):
#         # self.fft()
#         # self.fttttt()
#         self.draw_spectrogram(self.original_signal, self.grSpec)
#         # self.play_sound()
#         # self.spec_plot.setLimits(xMin=0, xMax=self.t[-1], yMin=0, yMax=self.f[-1])
#         _thread.start_new_thread(self.play_sound, ())
#         self.grMain.plotItem.setYRange(min(self.original_signal.real)*2, max(self.original_signal.real)*2)
#         self.grMain.plot(self.Time, self.original_signal.real, pen = self.pen)
#         self.grUpdatedMain.plotItem.setYRange(min(self.original_signal.real)*2, max(self.original_signal.real)*2)
#         self.updated_graphs()
#         self.update()
    
#     def pause(self):
#         self.timer.stop()
#         # self.iterator = 0
        
        
#     def toggle(self):
#         if self.actionPlay.isChecked():
#             self.play()
#         else:
#             self.pause()
            
#     def stop(self):
#         self.timer.stop()
#         self.grSpec.clear()
#         self.grUpdatedSpec.clear()
#         self.grUpdatedMain.plotItem.clearPlots()
#         self.grMain.plotItem.clearPlots()
#         self.actionPlay.setChecked(False)
#         self.iterator = 0
        
#     # def fft(self):
#     #     N = int(len(self.signal))
#     #     self.spectrum = fft(self.signal)
#     #     self.spectrum = 2.0/N * np.abs(self.spectrum[0:N//2])
#     #     self.freq = fftfreq(len(self.spectrum), 1/self.samplerate)
            
#     # def exporting_to_csv(self):
#     #     data = {'freq': self.freq,'spectrum': self.spectrum } #list(np.arange(0,len(self.freq),1))
#     #     df = pd.DataFrame(data, columns= ['freq', 'spectrum'])
#     #     name = QFileDialog.getSaveFileName(self, 'Save File')
#     #     df.to_csv (str(name[0]), index = False, header=True)
        
#     def draw_spectrogram(self, plotted_signal, graph):
#         graph.clear()
#         f, t, Sxx = signal.spectrogram(plotted_signal.real, self.samplerate)
#         # Interpret image data as row-major instead of col-major
#         pg.setConfigOptions(imageAxisOrder='row-major')
#         pg.mkQApp()
#         spec_plot = graph.addPlot()
#         img = pg.ImageItem()
#         spec_plot.addItem(img)
#         hist = pg.HistogramLUTItem() # histogram to control the gradient of the image
#         hist.setImageItem(img)
#         graph.addItem(hist)
#         # self.grSpec.show()
#         hist.setLevels(np.min(Sxx), np.max(Sxx))
#         img.setImage(Sxx) # Sxx: amplitude for each pixel
#         img.scale(t[-1]/np.size(Sxx, axis=1),
#                   f[-1]/np.size(Sxx, axis=0))
        
#         spec_plot.setLimits(xMin=0, xMax=t[-1], yMin=0, yMax=f[-1])
#         spec_plot.setLabel('bottom', "Time", units='s')
#         spec_plot.setLabel('left', "Frequency", units='Hz')
#         hist.gradient.restoreState({'ticks': [(0.0, (0, 0, 0, 255)), (0.01, (32, 0, 129, 255)),
#                                             (0.8, (255, 255, 0, 255)), (0.5, (115, 15, 255, 255)),
#                                             (1.0, (255, 255, 255, 255))], 'mode': 'rgb'})
#         #(32, 0, 129, 255)
    
    
# def main():
#     app = QApplication(sys.argv)
#     window = MainApp()
#     window.show()
#     app.exec_()
# if __name__ == '__main__':
#     main()




# import sys
# from os import path
# from PyQt5 import QtGui, QtCore
# from PyQt5.QtGui import *
# from PyQt5.QtCore import * 
# from PyQt5.uic import loadUiType
# import pyqtgraph as pg
# from PIL import Image
# import numpy as np

# FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui.ui"))
# class DrawImage(QMainWindow, FORM_CLASS): 
#     def __init__(self, parent=None):
#         super(QMainWindow, self).__init__(parent)
        
#         self.setupUi(self)

#         self.setWindowTitle('Select Window')
#         self.local_image = QImage('hgjart.JPG')
        
#         self.image = Image.open('hgjart.JPG')
#         self.frame = np.asarray(self.image)
#         self.local_image = pg.ImageItem(self.frame)

#         self.local_grview = QGraphicsView()
#         self.setCentralWidget( self.local_grview )

#         self.local_scene = QGraphicsScene()
#         # self.local_scene = self.grUpdatedSpec.addPlot()
        
#         # self.image_format = self.local_image.format()
#         # self.pixMapItem = self.local_scene.addPixmap( QPixmap(self.local_image) )
#         # self.pixMapItem = QGraphicsPixmapItem(QPixmap(self.local_image))
#         # self.local_scene.PlotWidget(self.pixMapItem)
#         self.local_scene.addItem(self.local_image)

#         self.local_grview.setScene( self.local_scene )

#         # self.local_grview.mousePressEvent = self.pixelSelect
#         self.position = self.pos
#         self.local_grview.mousePressEvent = self.mouseMoved(self.position)

#     def pixelSelect( self, event ):
#         print('hello')
#         position = QPoint( event.pos().x(),  event.pos().y())
#         color = QColor.fromRgb(self.local_image.pixel( position ) )
#         if color.isValid():
#             rgbColor = '('+str(color.red())+','+str(color.green())+','+str(color.blue())+','+str(color.alpha())+')'
#             self.setWindowTitle( 'Pixel position = (' + str( event.pos().x() ) + ' , ' + str( event.pos().y() )+ ') - Value (R,G,B,A)= ' + rgbColor)
#         else:
#             self.setWindowTitle( 'Pixel position = (' + str( event.pos().x() ) + ' , ' + str( event.pos().y() )+ ') - color not valid')

#     def pos( self, event ):
#             print('hello')
#             position = QPoint( event.pos().x(),  event.pos().y())
#             return position
    
#     def mouseMoved(self, viewPos):
    
#         data = self.local_image.image  # or use a self.data member
#         nRows, nCols, _ = data.shape 
    
#         scenePos = self.image.getImageItem().mapFromScene(viewPos)
#         row, col = int(scenePos.y()), int(scenePos.x())
    
#         if (0 <= row < nRows) and (0 <= col < nCols):
#             value = data[row, col]
#             print("pos = ({:d}, {:d}), value = {!r}".format(row, col, value))
#         else:
#             print("no data at cursor")
        
        
# def main():
#     app = QtGui.QApplication(sys.argv)
#     form = DrawImage()
#     form.show()
#     app.exec_()

# if __name__ == '__main__':
#     main()






##################################################
##################### WORKING ####################
####### adds an image inside an image 
####### item and adds it to 
####### a pg.GraphicsLayoutWidget()
##################################################
##################################################

# import numpy as np
# from PyQt5 import QtWidgets
# import pyqtgraph as pg
# from PIL import Image


# class TestImage(QtWidgets.QMainWindow):

#     def __init__(self):
#         super().__init__()

#         # Basic UI layout
#         self.statusbar = QtWidgets.QStatusBar(self)
#         self.setStatusBar(self.statusbar)
#         self.glw = pg.GraphicsLayoutWidget()
#         self.setCentralWidget(self.glw)

#         # Make image plot
#         self.p1 = self.glw.addPlot()
#         self.p1.getViewBox().setAspectLocked()
#         # Draw axes and ticks above image/data
#         [ self.p1.getAxis(ax).setZValue(10) for ax in self.p1.axes ]
#         self.data = np.random.rand(120, 100)
        
#         self.image = Image.open('hgjart.JPG')
#         self.frame = np.asarray(self.image)
#         # self.frame = np.random.rand(120, 100)
#         self.local_image = pg.ImageItem(self.frame)
        
#         self.img = pg.ImageItem(self.data)
#         self.p1.addItem(self.local_image)
#         # Centre axis ticks on pixel
#         self.img.setPos(-0.5, -0.5)

#         # Swap commented lines to choose between hover or click events
#         self.p1.scene().sigMouseMoved.connect(self.mouseMovedEvent)
#         #self.p1.scene().sigMouseClicked.connect(self.mouseClickedEvent)

#     def mouseClickedEvent(self, event):
#         self.mouseMovedEvent(event.pos())

#     def mouseMovedEvent(self, pos):
#         # Check if event is inside image, and convert from screen/pixels to image xy indicies
#         if self.p1.sceneBoundingRect().contains(pos):
#             mousePoint = self.p1.getViewBox().mapSceneToView(pos)
#             x_i = round(mousePoint.x())
#             y_i = round(mousePoint.y())
#             if x_i > 0 and x_i < self.frame.shape[0] and y_i > 0 and y_i < self.frame.shape[1]:
#                 self.statusbar.showMessage("({}, {}) = {:0.2f}".format(x_i, y_i, self.frame[x_i, y_i,0]))
#                 return
#         self.statusbar.clearMessage()

# def main():
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     mainwindow = TestImage()
#     mainwindow.show()
#     sys.exit(app.exec_())

# if __name__ == '__main__':
#     main()






########################################
## Testing with exp (phase functions)
########################################

# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image 
# import scipy.fftpack as spfft
# import cv2

# # Generate a phantom image
# phantom = np.array([[0, 0, 0, 0, 0],
#                     [0, 1, 1, 1, 0],
#                     [0, 1, 0, 1, 0],
#                     [0, 1, 1, 1, 0],
#                     [0, 0, 0, 0, 0]])

# # phantom = np.array([[1, 0, 0],
# #                     [0, 1, 0],
# #                     [0, 0, 1]])


# # image = Image.open('sword.JPG').convert('L')
# # phantom = np.array(image.getdata())
# image =cv2.imread("sword2.JPG",0)
# phantom = np.array(image)
# # phantom = np.reshape(phantom, (236, -1))
# print(phantom.shape[0])
# # frame = np.asarray(image)

# # Define the sequence parameters
# TR = 1000  # repetition time
# TE = 30  # echo time
# alpha = np.deg2rad(90)  # flip angle
# gamma = 42.58  # gyromagnetic ratio in MHz/T for Hydrogen
# FOV = 0.1  # field of view in meters
# Nx = phantom.shape[0]
# Ny = phantom.shape[1]  # matrix size
# dx = (2 * np.pi) / Nx
# dy = (2 * np.pi) / Ny #FOV / Nx  # voxel size
# # dx = FOV / Nx
# # dy = FOV / Ny

# # Apply the RF pulse
# Ry = np.array([[np.cos(alpha), 0, -np.sin(alpha)],
#                 [0, 1, 0],
#                 [np.sin(alpha), 0, np.cos(alpha)]])

# Rx = np.array([[1,0,0],
#                 [0, np.cos(alpha), -np.sin(alpha)],
#                 [0, np.sin(alpha), np.cos(alpha)]])


# # M = np.dot(Rx, phantom.T).T
# M = np.zeros((3,1,Nx*Ny))
# print(phantom.shape[0], phantom.shape[1], 'shape 0 and 1')
# rotated = np.zeros((3,1))

# for (x , y), spin in np.ndenumerate(phantom): 
#     # rotated = np.dot(np.array([x, y, spin]).T, Rx)
#     rotated[:,0] = np.matmul(Rx, np.array([0, 0, spin]))
#     rotated_spin = np.sqrt(rotated[0]**2 + rotated[1]**2 + rotated[2]**2)
#     M[:,:,x*y] = rotated
#     # print(rotated.T)
#     # print(M, 'M after')
    
# print(rotated.shape, 'shape of rotated')
# print(M.shape, 'shape of M before')
# M = np.reshape(M[2], (-1, phantom.shape[1])) # -1 tells the function this dimension is unknown and it calculates it for you, it will be 5 in our case

# print(M.shape, 'M.shape')
# print(M[41,20].shape, 'M.shape')

# # Apply the gradient fields and acquire k-space data
# k_space = np.zeros((Nx, Ny), dtype=np.float) # complex64
# for ky in range(Ny):
#     for kx in range(Nx):
#         # phase_y = np.exp(1j * gamma * 2 * np.pi * ky  * np.arange(Ny) / Ny)
#         # phase_x = np.exp(1j * gamma * 2 * np.pi * kx  * np.arange(Nx) / Nx)
        
#         phase_y = np.exp(1j * gamma * ky * dy * np.arange(Ny) )
#         phase_x = np.exp(1j * gamma * kx * dx * np.arange(Nx) )
        
#         phase = np.outer(phase_x, phase_y)
        
#         k_space[kx, ky] = np.sum(M * phase)
#         # print('K: ', ky, kx)

# # k_space = spfft.fftshift(k_space)
# # Inverse Fourier Transform
# # image = np.fft.fft2(phantom)
# image = np.fft.ifft(k_space).real

# # Display the results
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5)) # , figsize=(10, 5)
# ax1.imshow(phantom, cmap='gray')
# ax1.set_title('Phantom')
# ax2.imshow(image, cmap='gray')
# ax2.set_title('Reconstructed')
# ax3.imshow(k_space, cmap='gray')
# ax3.set_title('K-Space')
# # ax2.set_title('K-Space')
# plt.show()




# ####################################
# # Here we'll try rotating each row and each column 
# # and then sum up all the Xs, Ys, Zs for each voxel
# # sbrna ya rab 
# ####################################


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import scipy.fftpack as spfft
import cv2

# Generate a phantom image
phantom = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]])

# phantom = np.array([[1, 0, 0],
#                     [0, 1, 0],
#                     [0, 0, 1]])


# image = Image.open('sword.JPG').convert('L')
# phantom = np.array(image.getdata())
image =cv2.imread("sword2_45p.JPG",0)
phantom = np.array(image)
# phantom = np.reshape(phantom, (236, -1))
print(phantom.shape[0])
# frame = np.asarray(image)

# Define the sequence parameters
TR = 1000  # repetition time
TE = 30  # echo time
alpha = np.deg2rad(90)  # flip angle
gamma = 42.58  # gyromagnetic ratio in MHz/T for Hydrogen
FOV = 0.1  # field of view in meters
Ny = phantom.shape[0]
Nx = phantom.shape[1]  # matrix size
dy = (2 * np.pi) / Nx
dx = (2 * np.pi) / Ny #FOV / Nx  # voxel size
# dx = FOV / Nx
# dy = FOV / Ny

# Apply the RF pulse
Ry = np.array([[np.cos(alpha), 0, np.sin(alpha)],
                [0, 1, 0],
                [-np.sin(alpha), 0, np.cos(alpha)]])

Rx = np.array([[1,0,0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]])

Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0,0,1],])


def rotate_about_z(angle):
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0,0,1],])
    return Rz


# M = np.dot(Rx, phantom.T).T
M = np.zeros((3,1, Ny, Nx))
print(phantom.shape[0], phantom.shape[1], 'shape 0 and 1')
rotated = np.zeros((3,1))

# Apply the gradient fields and acquire k-space data
k_space = np.zeros((Ny, Nx), dtype=np.complex64) # complex64
gradient_image = np.zeros((Nx, Ny), dtype=np.float64)
multiplier = np.zeros((3, 3, Nx), dtype = np.int32)

pixel_vector = np.zeros((3,1))
M_rotated = np.zeros((3, 1, Ny, Nx))

kx = np.linspace(-Nx//2, Nx//2-1, Nx)/Nx
ky = np.linspace(-Ny//2, Ny//2-1, Ny)/Ny

for (y , x), spin in np.ndenumerate(phantom): 
    # rotated = np.dot(np.array([x, y, spin]).T, Rx)
    rotated = np.dot(Rx, np.array([0, 0, spin]))
    
    pixel_vector = np.dot(rotate_about_z(y*dy), rotated)
    pixel_vector = np.dot(rotate_about_z(x*dx), pixel_vector)
    rotated_spin = np.sum(pixel_vector)
    # rotated_spin = np.sqrt(pixel_vector[0]**2 + pixel_vector[1]**2 + pixel_vector[2]**2)
    
    # phase_y = np.exp(1j * gamma * y * dy * np.arange(Ny)[y] )
    # phase_x = np.exp(1j * gamma * x * dx * np.arange(Nx)[x] )
    
    # phase = np.outer(phase_x, phase_y)
    
    # k_space[y, x] = np.sum(pixel_vector * phase)

    # M[:,:,x*y] = rotated
    k_space[y, x] = rotated_spin
    print(rotated)
    # print(M, 'M after')
    
# print(rotated.shape, 'shape of rotated')
# print(M.shape, 'shape of M before')
# print(M, 'M')
# M = np.reshape(M, (3, 1, phantom.shape[0], phantom.shape[1])) # -1 tells the function this dimension is unknown and it calculates it for you, it will be 5 in our case

# print(M.shape, 'M.shape')

# # Apply the gradient fields and acquire k-space data
# k_space = np.zeros((Nx, Ny), dtype=np.complex64) # complex64
# gradient_image = np.zeros((Nx, Ny), dtype=np.float64)
# multiplier = np.zeros((3, 3, Nx), dtype = np.int32)

# # for i in range(Nx):
# #     multiplier[:, :, i] = Rz
    
# print(multiplier.shape)
# pixel_vector = np.zeros((3,1))
# M_rotated = np.zeros((3,1,Nx*Ny))

for ky in range(Ny):
#     # gradient_image[ky,:] = np.einsum('ij, ij->i', M[ky, :], multiplier)
#     # faaaaaaaks.. hndrbhom kolhm fel loop bta3t el kx.. nedrb el row kolo
#     # fe Rz bta3t Gy w ba3dha nedrab kol wa7da fel el Gx bta3tha
#     # bs msh dlw2ty b2a ana 3ayz anam sa3ayaa
#     # print(gradient_image)
    for kx in range(Nx):
        
        pass
#         pixel_vector = np.matmul(Rz, M[:,:, ky, kx])
        
#         # gradient_image = np.matmul(M[].T, Rx)
#         # phase_y = np.exp(1j * gamma * 2 * np.pi * ky  * np.arange(Ny) / Ny)
#         # phase_x = np.exp(1j * gamma * 2 * np.pi * kx  * np.arange(Nx) / Nx)
        
#         # phase_y = np.exp(1j * gamma * ky * dy * np.arange(Ny) )
#         # phase_x = np.exp(1j * gamma * kx * dx * np.arange(Nx) )
        
#         # phase = np.outer(phase_x, phase_y)
        
#         # k_space[kx, ky] = np.sum(M * phase)
#         # print('K: ', kx, ky)


# k_space = spfft.fftshift(k_space)
# Inverse Fourier Transform
# image = np.fft.fft2(phantom)
image = np.fft.ifft2(k_space).real

# Display the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5)) # , figsize=(10, 5)
ax1.imshow(phantom, cmap='gray')
ax1.set_title('Phantom')
ax2.imshow(image, cmap='gray')
ax2.set_title('Reconstructed')
ax3.imshow(np.log(np.abs(spfft.fftshift(k_space))), cmap='gray')
ax3.set_title('K-Space')
# ax2.set_title('K-Space')
plt.show()
