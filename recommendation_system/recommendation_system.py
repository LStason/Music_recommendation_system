# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import wx
import wx.media
import wx.lib.buttons as buttons
import os
import csv
import librosa
from yaafelib import *
import numpy as np
from mutagen.mp3 import MP3
from sklearn import preprocessing, cross_validation
from sklearn import svm, metrics, linear_model
from sklearn.preprocessing import Imputer
from sklearn import cross_validation, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import threading
import Queue
import time
import scipy
import sys
import subprocess
import math

dirName = "/media/sf_Shared/Diplom/"	#путь к папке с программой
bitmapDir = os.path.join(dirName, 'bitmaps')	#путь к изображениям
path_train_csv = '/media/sf_Shared/Diplom/train_data.csv'	#путь к CSV файлу с обучающим набором данных для классификатора
path_music_database = '/media/sf_Shared/Diplom/music_database.csv'	#путь к CSV файлу музыкальной базы
path_playlist = '/home/stas/Рабочий стол/playlist.m3u'	#путь к плейлисту

class Recommendation_System(wx.Frame):
    #------------------------------------------------------------------------------
    def __init__(self, *args, **kw):
        super(Recommendation_System, self).__init__(*args, **kw)
        self.train_data = []
        self.train_data = self.read_csv(path_train_csv)
        self.music_database = self.read_csv(path_music_database)
        self.Training()
        self.classifiers = ['KNN', 'SVM', 'Random Forest', 'Logistic Regression']
        self.metrics = ['Euclidean','Cosine','Pearson']
        self.InitUI()
        self.mediaPlayer = wx.media.MediaCtrl(self, style=wx.SIMPLE_BORDER)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimer)
        self.timer.Start(100)
        self.timer2 = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer2, self.timer2)
        self.timer3 = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer3, self.timer3)
    #------------------------------------------------------------------------------
    def InitUI(self):
        font1 = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        font2 = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        font3 = wx.Font(8, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        self.count = 0
        self.task_range = 0

        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
        self.SetSize((500, 720))
        self.SetTitle('Music recommendation system')

        pnl = wx.Panel(self)

        self.header0 = wx.StaticText(pnl, label='Information', pos=(40,15))
        self.header0.SetFont(font2)
        self.text0 = wx.StaticText(pnl, label='Size of the music database:', pos=(20,40))
        self.text0.SetFont(font1)
        self.music_database_size = wx.StaticText(pnl, label=str(len(self.music_database)), pos=(255, 40))
        self.music_database_size.SetFont(font2)
        self.text1 = wx.StaticText(pnl, label='Size of the train data:', pos=(20,60))
        self.text1.SetFont(font1)
        self.train_data_size = wx.StaticText(pnl, label=str(len(self.train_data)), pos=(205, 60))
        self.train_data_size.SetFont(font2)
        self.text2 = wx.StaticText(pnl, label='Number of genres:', pos=(20,80))
        self.text2.SetFont(font1)
        self.number_of_genres = wx.StaticText(pnl, label=str(len(self.genres)), pos=(180, 80))
        self.number_of_genres.SetFont(font2)

        self.line1 = wx.StaticLine(pnl, pos=(15, 105), size=(470,2))

        self.header1 = wx.StaticText(pnl, label='Add data to music database', pos=(40,110))
        self.header1.SetFont(font2)
        self.text3 = wx.StaticText(pnl, label='New data:', pos=(40,140))
        self.text3.SetFont(font1)
        self.tc1 = wx.TextCtrl(pnl, pos=(150,135), size = (300, 30))
        self.tc1.SetFont(font3)
        broseButton1 = wx.Button(pnl, wx.ID_ANY, '...', pos=(460, 135), size=(30,30))
        addButton = wx.Button(pnl, wx.ID_ANY, 'Add', pos=(405, 175))
        saveButton = wx.Button(pnl, wx.ID_ANY, 'Save', pos=(405, 205))
        self.gauge1 = wx.Gauge(pnl, size=(245, 25), pos=(150,175))
        self.task_text1 = wx.StaticText(pnl, label='', pos=(150,205))

        self.line2 = wx.StaticLine(pnl, pos=(15, 240), size=(470,2))

        self.header2 = wx.StaticText(pnl, label='Genre classification', pos=(40,245))
        self.header2.SetFont(font2)
        self.text4 = wx.StaticText(pnl, label='Load train data:', pos=(20,280))
        self.text4.SetFont(font1)
        self.tc2 = wx.TextCtrl(pnl, pos=(150,275), size = (300, 30))
        self.tc2.SetFont(font3)
        broseButton2 = wx.Button(pnl, wx.ID_ANY, '...', pos=(460, 275), size=(30,30))
        self.gauge2 = wx.Gauge(pnl, size=(245, 25), pos=(150,310))
        addButton2 = wx.Button(pnl, wx.ID_ANY, 'Add', pos=(405, 310))
        self.task_text2 = wx.StaticText(pnl, label='', pos=(150,340))
        self.text5 = wx.StaticText(pnl, label='Load CSV file:', pos=(30,375))
        self.text5.SetFont(font1)
        self.tc3 = wx.TextCtrl(pnl, pos=(150,370), size = (300, 30))
        self.tc3.SetFont(font3)
        broseButton3 = wx.Button(pnl, wx.ID_ANY, '...', pos=(460, 370), size=(30,30))
        self.text6 = wx.StaticText(pnl, label='Classifier:', pos=(60,410))
        self.text6.SetFont(font1)
        self.cb = wx.ComboBox(pnl, pos=(150, 405), choices=self.classifiers, style=wx.CB_READONLY)
        checkButton = wx.Button(pnl, wx.ID_ANY, 'Check', pos=(405, 405))
        self.text7 = wx.StaticText(pnl, label='Accuracy:', pos=(60,440))
        self.text7.SetFont(font1)
        self.text8 = wx.StaticText(pnl, label='', pos=(150,440))
        self.text8.SetFont(font2)

        self.line3 = wx.StaticLine(pnl, pos=(15, 465), size=(470,2))

        self.header3 = wx.StaticText(pnl, label='Recommendation', pos=(40,470))
        self.header3.SetFont(font2)
        self.text9 = wx.StaticText(pnl, label='Load audio file:', pos=(20,500))
        self.text9.SetFont(font1)
        self.tc4 = wx.TextCtrl(pnl, pos=(150,495), size = (300, 30))
        self.tc4.SetFont(font3)
        broseButton4 = wx.Button(pnl, wx.ID_ANY, '...', pos=(460, 495), size=(30,30))
        loadButton = wx.Button(pnl, wx.ID_ANY, 'Load', pos=(405, 530))
        self.playbackSlider = wx.Slider(pnl, value=0, pos=(90,535), size=(300, -1), style=wx.SL_HORIZONTAL)
        img = wx.Bitmap(os.path.join(bitmapDir, "player_play.png"))
        self.playPauseBtn = buttons.GenBitmapToggleButton(pnl, bitmap=img, name="play", pos=(30,530))
        self.playPauseBtn.Enable(False)
        img = wx.Bitmap(os.path.join(bitmapDir, "player_pause.png"))
        self.playPauseBtn.SetBitmapSelected(img)
        self.playPauseBtn.SetInitialSize()
        classificateButton = wx.Button(pnl, wx.ID_ANY, 'Define genre', pos=(30, 585))
        self.text10 = wx.StaticText(pnl, label='', pos=(150,590))
        self.text10.SetFont(font1)
        self.text11 = wx.StaticText(pnl, label='Number of songs:', pos=(20,630))
        self.text11.SetFont(font1)
        self.tc5 = wx.TextCtrl(pnl, pos=(170,625), size = (60, 30))
        self.text12 = wx.StaticText(pnl, label='Metric:', pos=(235,630))
        self.text12.SetFont(font1)
        self.cb2 = wx.ComboBox(pnl, pos=(295, 625), choices=self.metrics, style=wx.CB_READONLY)
        recommendationButton = wx.Button(pnl, wx.ID_ANY, 'Get recommendation', pos=(30, 670))
        exitButton = wx.Button(pnl, wx.ID_ANY, 'Exit', pos=(400, 670))

        self.Bind(wx.EVT_BUTTON, self.DirBrose1, id=broseButton1.GetId())
        self.Bind(wx.EVT_BUTTON, self.AddMusicDatabase, id=addButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.SaveData, id=saveButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.DirBrose2, id=broseButton2.GetId())
        self.Bind(wx.EVT_BUTTON, self.AddTrainData, id=addButton2.GetId())
        self.Bind(wx.EVT_BUTTON, self.FileBrose1, id=broseButton3.GetId())
        self.Bind(wx.EVT_BUTTON, self.CheckTrainData, id=checkButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.FileBrose2, id=broseButton4.GetId())
        self.Bind(wx.EVT_BUTTON, self.loadMusic, id=loadButton.GetId())
        self.Bind(wx.EVT_SLIDER, self.onSeek, self.playbackSlider)
        self.Bind(wx.EVT_BUTTON,  self.OnExit, id=exitButton.GetId())
        self.playPauseBtn.Bind(wx.EVT_BUTTON, self.onPlay)
        self.Bind(wx.EVT_BUTTON, self.DefineGenreThread, id=classificateButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.GetRecommendation, id=recommendationButton.GetId())

        self.Centre()
        self.Show(True)
    #------------------------------------------------------------------------------
    def Training(self):
        self.genres = self.get_genres(self.train_data)
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.genres)

        self.x_data = np.array([track['features'] for track in self.train_data])
        self.x_scale_data = preprocessing.scale(self.x_data)
        self.imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.x_scale_data = self.imp.fit_transform(self.x_scale_data)
        self.x_target = np.array([int(self.le.transform([track['genre']])) for track in self.train_data])

        self.model_svc = svm.SVC(kernel='rbf', C=6)
        self.model_rfc = RandomForestClassifier(n_estimators=74)
        self.model_knc = KNeighborsClassifier(n_neighbors = 1)
        self.model_lr = LogisticRegression(penalty='l1', C=0.15)

        self.model_knc.fit(self.x_scale_data, self.x_target)
        self.model_svc.fit(self.x_scale_data, self.x_target)
        self.model_rfc.fit(self.x_scale_data, self.x_target)
        self.model_lr.fit(self.x_scale_data, self.x_target)
    #------------------------------------------------------------------------------
    def OnExit(self, event):
        self.Close()
    #------------------------------------------------------------------------------
    def DirBrose1(self, event):
        dlg = wx.DirDialog(self, "Choose a directory:", defaultPath="/media/sf_Shared/Diplom/Files/",
                           style=wx.DD_DEFAULT_STYLE
                           #| wx.DD_DIR_MUST_EXIST
                           #| wx.DD_CHANGE_DIR
                           )
        if dlg.ShowModal() == wx.ID_OK:
            self.tc1.SetValue(dlg.GetPath())
            self.gauge1.SetValue(0)
            self.task_text1.SetLabel('Task to be done')
        dlg.Destroy()
    #------------------------------------------------------------------------------
    def DirBrose2(self, event):
        dlg = wx.DirDialog(self, "Choose a directory:", defaultPath="/media/sf_Shared/Diplom/Files/",
                           style=wx.DD_DEFAULT_STYLE
                           #| wx.DD_DIR_MUST_EXIST
                           #| wx.DD_CHANGE_DIR
                           )
        if dlg.ShowModal() == wx.ID_OK:
            self.tc2.SetValue(dlg.GetPath())
            self.gauge2.SetValue(0)
            self.task_text2.SetLabel('Task to be done')
        dlg.Destroy()
    #------------------------------------------------------------------------------
    def FileBrose1(self, event):
        stl=wx.OPEN
        filedia=wx.FileDialog(self,'Choose a file',"/media/sf_Shared/Diplom/", "", "CSV files (*.csv)|*.csv")
        if filedia.ShowModal()==wx.ID_OK:
            self.tc3.SetValue(filedia.GetPath())
        filedia.Destroy()
    #------------------------------------------------------------------------------
    def FileBrose2(self, event):
        stl=wx.OPEN
        filedia=wx.FileDialog(self,'Choose a file', "/media/sf_Shared/Diplom/Files")
        if filedia.ShowModal()==wx.ID_OK:
            self.tc4.SetValue(filedia.GetPath())
            self.text10.SetLabel('')
        filedia.Destroy()
    #------------------------------------------------------------------------------
    def AddMusicDatabase(self, event):
        path = self.tc1.GetValue()
        if path == '':
            wx.MessageBox("Folder is not selected!",
                          "ERROR",
                          wx.ICON_ERROR | wx.OK)
        else:
            self.count = 0
            self.new_data = []
            self.new_data = self.generate_md_track_list(path)
            self.task_range = len(self.new_data)
            self.gauge1.SetRange(self.task_range)
            self.timer2.Start(100)
            self.task_text1.SetLabel('Task in Progress')
            self.th1 = threading.Thread(target=self.features_extractor, args = ([self.new_data]))
            self.th1.start()
    #------------------------------------------------------------------------------
    def OnTimer2(self, event):
        self.gauge1.SetValue(self.count)
        if self.count == self.task_range:
            self.timer2.Stop()
            self.genres_prediction(self.new_data, self.x_data, self.model_svc)
            self.artist_definition(self.new_data)
            self.task_text1.SetLabel('Task Completed!')
            self.th1.join()
            self.music_database = self.music_database + self.new_data
            self.music_database_size.SetLabel(str(len(self.music_database)))
    #------------------------------------------------------------------------------
    def SaveData(self, event):
        self.write_csv(path_music_database, self.music_database)
        self.task_text1.SetLabel('Data saved successfully!')
    #------------------------------------------------------------------------------
    def AddTrainData(self, event):
        path = self.tc2.GetValue()
        if path == '':
            wx.MessageBox("Folder is not selected!",
                          "ERROR",
                          wx.ICON_ERROR | wx.OK)
        else:
            self.count = 0
            self.train_data = []
            self.train_data = self.generate_track_list(path, self.train_data)
            self.task_range = len(self.train_data)
            self.gauge2.SetRange(self.task_range)
            self.timer3.Start(100)
            self.task_text2.SetLabel('Task in Progress')
            self.th2 = threading.Thread(target=self.features_extractor, args = ([self.train_data]))
            self.th2.start()
    #------------------------------------------------------------------------------
    def OnTimer3(self, event):
        path = self.tc2.GetValue()
        self.gauge2.SetValue(self.count)
        if self.count == self.task_range:
            self.timer3.Stop()
            self.task_text2.SetLabel('Task Completed')
            new_path = os.path.join(dirName, path.split('/')[-1] + '.csv')
            self.write_csv(new_path, self.train_data)
            self.tc3.SetValue(new_path)
            self.th2.join()
    #------------------------------------------------------------------------------
    def CheckTrainData(self, event):
        path_train_data = self.tc3.GetValue()
        if path_train_data == '':
            wx.MessageBox("CSV file is not selected!",
                          "ERROR",
                          wx.ICON_ERROR | wx.OK)
        else:
            self.train_data = []
            self.train_data = self.read_csv(self.tc3.GetValue())
            self.Training()
            self.train_data_size.SetLabel(str(len(self.train_data)))
            self.number_of_genres.SetLabel(str(len(self.genres)))

            classifier = self.cb.GetValue()

            if classifier == '':
                wx.MessageBox("Classifier is not selected!",
                              "ERROR",
                              wx.ICON_ERROR | wx.OK)
            elif classifier == 'KNN':
                scores = cross_validation.cross_val_score(self.model_knc, self.x_scale_data, self.x_target, cv = 10)
                self.text8.SetLabel(str(scores.mean()))
            elif classifier == 'SVM':
                scores = cross_validation.cross_val_score(self.model_svc, self.x_scale_data, self.x_target, cv = 10)
                self.text8.SetLabel(str(scores.mean()))
            elif classifier == 'Random Forest':
                scores = cross_validation.cross_val_score(self.model_rfc, self.x_scale_data, self.x_target, cv = 10)
                self.text8.SetLabel(str(scores.mean()))
            elif classifier == 'Logistic Regression':
                scores = cross_validation.cross_val_score(self.model_lr, self.x_scale_data, self.x_target, cv = 10)
                self.text8.SetLabel(str(scores.mean()))
    #------------------------------------------------------------------------------
    def DefineGenreThread(self, event):
        self.th3 = threading.Thread(target=self.DefineGenre)
        self.th3.start()
        self.th3.join()
    #------------------------------------------------------------------------------
    def DefineGenre(self):
        path = self.tc4.GetValue()
        if path == '':
            wx.MessageBox("Audio file is not selected!",
                          "ERROR",
                          wx.ICON_ERROR | wx.OK)
        else:
            y_data = np.array(self.features(path))
            new_x_data = np.vstack((self.x_data, y_data))
            new_x_scale_data = preprocessing.scale(new_x_data)
            new_x_scale_data = self.imp.fit_transform(new_x_scale_data)
            y_scale_data = new_x_scale_data[-1]
            y_result_svc = self.model_svc.predict(y_scale_data)
            self.text10.SetLabel(str(self.le.inverse_transform(int(y_result_svc))))
    #------------------------------------------------------------------------------
    def GetRecommendationThread(self, event):
        self.th4 = threading.Thread(target=self.GetRecommendation)
        self.th4.start()
        self.th4.join()
    #------------------------------------------------------------------------------
    def GetRecommendation(self, event):
        path = self.tc4.GetValue()
        tracks_number = self.tc5.GetValue()
        if tracks_number != '':
            tracks_number = int(tracks_number)
        metric = self.cb2.GetValue()

        if path == '':
            wx.MessageBox("Audio file is not selected!",
                          "ERROR",
                          wx.ICON_ERROR | wx.OK)
        elif tracks_number == '' or tracks_number < 1:
            wx.MessageBox("Incorrect number of songs!",
                          "ERROR",
                          wx.ICON_ERROR | wx.OK)
        elif metric == '':
            wx.MessageBox("Metric is not selected!",
                          "ERROR",
                          wx.ICON_ERROR | wx.OK)
        else:
            track_features = np.array(self.features(path))
            new_x_data = np.vstack((self.x_data, track_features))
            new_x_scale_data = preprocessing.scale(new_x_data)
            new_x_scale_data = self.imp.fit_transform(new_x_scale_data)
            track_scale_features = new_x_scale_data[-1]
            track_genre = str(self.le.inverse_transform(int(self.model_svc.predict(track_scale_features))))
            track_artist = self.get_artist(path)
            music_database_features = preprocessing.scale(np.array([track['features'] for track in self.music_database]))
            candidates = []
            n = 0
            k = 0
            artist_filter = int(tracks_number/4)
            for i, track in enumerate(self.music_database):
                if track['genre'] == track_genre:
                    if metric == 'Euclidean':
                        temp = [self.euclidean_distance(track_scale_features, music_database_features[i]), track['name'], track['path'], track['artist']]
                        candidates.append(temp)
                    elif metric == 'Cosine':
                        temp = [self.cosine_similarity(track_scale_features, music_database_features[i]), track['name'], track['path'], track['artist']]
                        candidates.append(temp)
                    elif metric == 'Pearson':
                        temp = [self.correlation_coefficient(track_scale_features, music_database_features[i]), track['name'], track['path'], track['artist']]
                        candidates.append(temp)
            recommendation_list = []
            for candidate in sorted(candidates, reverse=True):
                if n == tracks_number:
                    break
                elif track_artist == candidate[3] and track_artist != 'None' and track_artist != '':
                    if k >= artist_filter:
                        continue
                    else:
                        recommendation_list.append(candidate[2])
                        k += 1
                        n += 1
                else:
                    recommendation_list.append(candidate[2])
                    n += 1
            self.create_playlist(path_playlist, recommendation_list)
            subprocess.call(['audacious', path_playlist])
    #------------------------------------------------------------------------------
    def loadMusic(self, event):
        if not self.mediaPlayer.Load(self.tc4.GetValue()):
            wx.MessageBox("Audio file is not selected!",
                          "ERROR",
                          wx.ICON_ERROR | wx.OK)
        else:
            self.mediaPlayer.SetInitialSize()
            self.playbackSlider.SetRange(0, self.mediaPlayer.Length())
            self.playPauseBtn.Enable(True)
    #------------------------------------------------------------------------------
    def onSeek(self, event):
        offset = self.playbackSlider.GetValue()
        self.mediaPlayer.Seek(offset)
    #------------------------------------------------------------------------------
    def onPlay(self, event):
        if not event.GetIsDown():
            self.onPause()
            return

        if not self.mediaPlayer.Play():
            wx.MessageBox("Unable to Play media : Unsupported format",
                          "ERROR",
                          wx.ICON_ERROR | wx.OK)
        else:
            self.mediaPlayer.SetInitialSize()
            self.playbackSlider.SetRange(0, self.mediaPlayer.Length())
        event.Skip()
    #------------------------------------------------------------------------------
    def onPause(self):
        self.mediaPlayer.Pause()
    #------------------------------------------------------------------------------
    def onTimer(self, event):
        offset = self.mediaPlayer.Tell()
        self.playbackSlider.SetValue(offset)
    #------------------------------------------------------------------------------
    def OnCloseWindow(self, event):
        dial = wx.MessageDialog(None, 'Are you sure to quit?', 'Question',
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
        ret = dial.ShowModal()
        if ret == wx.ID_YES:            
            self.Destroy()            
        else:
            event.Veto()
    #------------------------------------------------------------------------------
    def write_csv(self, path, track_list):
        with open(path, 'wb') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for track in track_list:
                row = []
                row.append(track['path'])
                row.append(track['name'])
                row.append(track['genre'])
                row.append(track['artist'])
                row.extend(track['features'])
                writer.writerow(row)
    #------------------------------------------------------------------------------
    def read_csv(self, path):
        return_data = []
        with open(path) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                d = {}
                d['path'] = row[0]
                d['name'] = row[1]
                d['genre'] = row[2]
                d['artist'] = row[3]
                d['features'] = row[4:]
                return_data.append(d)
        return return_data
    #------------------------------------------------------------------------------
    def get_genres(self, data):
        genre = set()
        for x in data:
            genre.add(x['genre'])
        return list(genre)
    #------------------------------------------------------------------------------
    def walk(self, path, track_list, genre = None):
        for name in os.listdir(path):
            d = {}
            new_path = os.path.join(path, name)
            if os.path.isfile(new_path):
                d['path'] = new_path
                d['name'] = name
                d['genre'] = genre
                d['artist'] = ''
                track_list.append(d)
            else:
                self.walk(new_path, track_list, name)
        return track_list
    #------------------------------------------------------------------------------
    def generate_track_list(self, path, track_list):
        return self.walk(path, track_list)
    #------------------------------------------------------------------------------
    def generate_md_track_list(self, path):
        files = librosa.util.find_files(path)
        track_list = []
        for f in files:
            d = {}
            d['path'] = f
            d['name'] = f.split('/')[-1]
            track_list.append(d)
        return track_list
    #------------------------------------------------------------------------------
    def calculate_statistics(self, data):
        statistics = []
        if data.ndim > 1:
            statistics.extend(np.mean(data, axis=0).tolist())
            statistics.extend(np.std(data, axis=0).tolist())
        else:
            statistics.append(np.mean(data))
            statistics.append(np.std(data))
        return statistics
    #------------------------------------------------------------------------------
    def features(self, filename):
        sampleRate = 22050
        fp = FeaturePlan(sample_rate=sampleRate)
        fp.addFeature('cdod: ComplexDomainOnsetDetection')
        fp.addFeature('energy: Energy')
        fp.addFeature('loudness: Loudness')
        fp.addFeature('obsi: OBSI')
        fp.addFeature('sfpb: SpectralFlatnessPerBand')
        fp.addFeature('sf: SpectralFlux')
        fp.addFeature('sroff: SpectralRolloff')
        fp.addFeature('sss: SpectralShapeStatistics')
        fp.addFeature('sv: SpectralVariation')
        fp.addFeature('ss: SpectralSlope')
        fp.addFeature('tss: TemporalShapeStatistics')
        fp.addFeature('zcr: ZCR')
        fp.addFeature('sflatness: SpectralFlatness')
        fp.addFeature('sd: SpectralDecrease')
        fp.addFeature('mfcc: MFCC')
        engine = Engine()
        engine.load(fp.getDataFlow())

        folder_size = os.path.getsize(filename)
        if folder_size*1.0/(1024*1024) > 20:
            audio, sr = librosa.load(filename, offset=30.0, duration=60.0, sr = sampleRate, dtype=np.float64)
        else:
            audio, sr = librosa.load(filename, sr = sampleRate, dtype=np.float64)
            seconds = librosa.get_duration(y=audio, sr=sr)
            if seconds > 120.0:
                audio = audio[int(len(audio)/2) - 60*sr:int(len(audio)/2)]

        f = []
        feats = engine.processAudio(np.array([audio]))
        f.extend(self.calculate_statistics(feats['mfcc']))
        f.extend(self.calculate_statistics(feats['cdod']))
        f.extend(self.calculate_statistics(feats['energy']))
        f.extend(self.calculate_statistics(feats['loudness']))
        f.extend(self.calculate_statistics(feats['obsi']))
        f.extend(self.calculate_statistics(feats['sfpb']))
        f.extend(self.calculate_statistics(feats['sf']))
        f.extend(self.calculate_statistics(feats['sroff']))
        f.extend(self.calculate_statistics(feats['sss']))
        f.extend(self.calculate_statistics(feats['sv']))
        f.extend(self.calculate_statistics(feats['ss']))
        f.extend(self.calculate_statistics(feats['tss']))
        f.extend(self.calculate_statistics(feats['zcr']))
        f.extend(self.calculate_statistics(feats['sflatness']))
        f.extend(self.calculate_statistics(feats['sd']))

        return f
    #------------------------------------------------------------------------------
    def features_extractor(self, track_list):
        self.count = 0
        for track in track_list:
            track['features'] = self.features(track['path'])
            self.count += 1
    #------------------------------------------------------------------------------
    def genres_prediction(self, track_list, x_data, model):
        data = np.array([track['features'] for track in track_list])
        new_x_data = np.vstack((x_data, data))
        new_x_scale_data = preprocessing.scale(new_x_data)
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        new_x_scale_data = imp.fit_transform(new_x_scale_data)
        scale_data = new_x_scale_data[len(x_data):]
        predictions = model.predict(scale_data)
        for i, track in enumerate(track_list):
            track['genre'] = str(self.le.inverse_transform(int(predictions[i])))
    #------------------------------------------------------------------------------
    def get_artist(self, filename):
        audio_format = filename.split('.')[-1]
        if audio_format != 'mp3':
            artist = 'None'
        else:
            try:
                audio = MP3(filename)
                if len(audio) == 0:
                    artist = 'None'
                elif 'TPE1' not in audio.tags:
                    artist = 'None'
                else:
                    artist = audio.tags['TPE1'].text[0].lower().encode('ascii', 'ignore').decode('ascii')

            except Exception as err:
                artist = 'None'
        return artist
    #------------------------------------------------------------------------------
    def artist_definition(self, track_list):
        for track in track_list:
            artist = self.get_artist(track['path'])
            track['artist'] = artist
    #------------------------------------------------------------------------------
    def euclidean_distance(self, a, b):
        result = 0
        for i in range(len(a)):
            result += (a[i] - b[i])**2
        return 1.0/(1 + math.sqrt(result))
    #------------------------------------------------------------------------------
    def cosine_similarity(self, a, b):
        def scalar(x, y):
            return reduce(lambda c, d: c + d, map(lambda c, d: c * d, x, y))
        if scalar(a, a) == 0 or scalar(b, b) == 0:
            return 0
        else:
            return (scalar(a, b)*1.0)/(math.sqrt(scalar(a, a))*math.sqrt(scalar(b, b))*1.0)
    #------------------------------------------------------------------------------
    def correlation_coefficient(self, x, y):
        n = len(x)
        vals = range(n)
        sumx = sum([float(x[i]) for i in vals])
        sumy = sum([float(y[i]) for i in vals])
        sumxSq = sum([x[i]**2.0 for i in vals])
        sumySq = sum([y[i]**2.0 for i in vals])
        pSum = sum([x[i]*y[i] for i in vals])
        num = pSum - (sumx*sumy/n)
        den = math.sqrt((sumxSq - pow(sumx, 2)/n)*(sumySq - pow(sumy, 2)/n))
        if den == 0:
            return 0
        return num/den
    #------------------------------------------------------------------------------
    def create_playlist(self, path, songs):
        _m3u = open(path, "w")
        for song in songs:
            _m3u.write(song + "\n")
        _m3u.close()
    #------------------------------------------------------------------------------

def main():
    ex = wx.App()
    Recommendation_System(None)
    ex.MainLoop()

if __name__ == '__main__':
    main()

