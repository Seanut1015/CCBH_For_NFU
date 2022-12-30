from PyQt5 import QtGui, QtWidgets
import speech_recognition as sr
from PyQt5.QtCore import Qt
from PIL import ImageGrab
import mediapipe as mp
import configparser
import numpy as np
import threading
import win32api
import win32con
import pyttsx3
import mouse
import math
import time
import cv2
import sys

program = 1
control = 1
catch = 0

def uirun():#設置基礎UI介面
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QWidget()
    window.setGeometry(0, 0, 200, 200)
    window.setStyleSheet("background:transparent")
    window.setAttribute(Qt.WA_TranslucentBackground)
    window.setWindowFlags(Qt.FramelessWindowHint)
    label = QtWidgets.QLabel(window)
    label.setText('Running') 
    font = QtGui.QFont()
    font.setFamily("微軟正黑體")
    font.setPointSize(12)
    label.setFont(font)
    label.setStyleSheet('''
    color:#0f0;
    ''')
    window.show() 
    sys.exit(app.exec_())

def takeCommand():#語音辨識程式
    global control
    global program
    global catch

    r=sr.Recognizer()

    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    rate = engine.getProperty('rate')           
    engine.setProperty('rate', rate-20)         
    engine.setProperty('volume',volume_f)  

    while program:
        with sr.Microphone(device_index = mic_id) as source:     
            r.adjust_for_ambient_noise(source,0.5)
            r.pause_threshold = 0.5
            #r.dynamic_energy_threshold = 1  
            audio = r.listen(source)
            try:
                query = r.recognize_google(audio, language='zh') 
                print(f"User said: {query}\n")   
                if query == '暫停':
                    control =0
                    engine.say('停止手勢操作')    
                elif query == '開始':
                    control =1
                    engine.say('啟動手勢操作') 
                elif query == '截圖':
                    catch = 1
                    time.sleep(0.1)
                    catch = 0
                    engine.say('OK') 
                elif query == '結束':
                    program = 0
                    break
                engine.runAndWait()
                
            except Exception as e:
                print('Command error')

def vector_2d_angle(v1, v2):#計算手指座標夾角
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

def hand_angle(hand_):#計算手指角度
    angle_list = []
    #thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    #middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    #ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    #pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

def hand_pos(finger_angle):#判斷手勢與功能
    f1 = finger_angle[0]   #大拇指角度
    f2 = finger_angle[1]   #食指角度
    f3 = finger_angle[2]   #中指角度
    f4 = finger_angle[3]   #無名指角度
    f5 = finger_angle[4]   #小拇指角度
    #手指角度 <50:伸直 >=50:捲縮
    #print(int(f1),int(f2),int(f3),int(f4),int(f5))
    if f1<50 and f2<50 and f3<50 and f4<50 and f5<50:#5
        return 5
    elif f1>=50 and f2<50 and f3>=50 and f4>=50 and f5>=50:#1
        return 1
    elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5>=50:#7
        return 7
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5<50:#4
        return 4
    elif f1<50 and f2<50 and f3<50 and f4>=50 and f5>=50:#8
        return 8
    elif f1>=50 and f2<50 and f3<50 and f4>=50 and f5>=50:#2
        return 2
    elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:#0
        return 0
    elif f1<50 and f2>=50 and f3>=50 and f4>=50 and f5<50:#6
        return 6
    else:
        return 99

def hand_draw():#繪製出手部模型
    mp_drawing.draw_landmarks(
        img,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

def list_updata(list,var):#更新串列
    list=np.append(list,var)
    list=np.delete(list,[0])
    return list

def pos_change(x,y):#座標轉換
    x = (x*camw)-camw_h
    y = (y*camh)-camh_h
    if x > w/2/sens:
        x = w/2/sens
    if x < -w/2/sens:
        x = -w/2/sens
    if y > h/2/sens:
        y = h/2/sens
    if y < -h/2/sens:
        y = -h/2/sens
    x = int(x*sens)+w/2
    y = int(y*sens)+h/2
    return x,y

#-------------------------------------載入config設定----------------------------------------
config = configparser.ConfigParser()
config.read('./config.ini')
#config.read('E:\Program\Github\CCBH_for_race\config.ini')
mainhand = int(config['user']['mainhand'])
unshake = int(config['user']['unshake'])
sens = float(config['user']['sens'])
mic_id = int(config['user']['mic'])
w = int(config['user']['display_w'])
h = int(config['user']['display_h'])
cam_id = int(config['user']['cam'])
camw = int(config['user']['cam_w'])
camh = int(config['user']['cam_h'])
volume = int(config['user']['volume'])
volume_f = float(volume/10)
window = int(config['user']['window'])
#-------------------------------------載入config設定----------------------------------------

#-------------------------------------初始設定----------------------------------------------
w_h = w/2
h_h = h/2
camw_h = camw/2
camh_h = camh/2

sre= threading.Thread(target=takeCommand)
sre.daemon = True 
sre.start()

ui= threading.Thread(target=uirun)
ui.daemon = True 
ui.start()

poselist_main=np.zeros((5),dtype=int)
poselist_sub=np.zeros((5),dtype=int)
xlist=np.zeros((unshake+1),dtype=int)
ylist=np.zeros((unshake+1),dtype=int)
xslist=np.zeros((unshake+1),dtype=int)
yslist=np.zeros((unshake+1),dtype=int)
xsdlist=np.zeros((unshake+1),dtype=int)
ysdlist=np.zeros((unshake+1),dtype=int)

pose_main = pose_sub = 0    #預設無手勢
pv_main=pv_sub=0            #紀錄上一個手勢
stop_wait_sec = 0 
pose_main_sure = pose_sub_sure =0
zoom = 0

xm2=ym2=xm8=ym8=xs2=ys2=xs8=ys8=0
xm2d=ym2d=xm8d=ym8d=xs2d=ys2d=xs8d=ys8d=0

mp_drawing = mp.solutions.drawing_utils          
mp_drawing_styles = mp.solutions.drawing_styles  
mp_hands = mp.solutions.hands   

cam=cv2.VideoCapture(cam_id,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, camw)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,camh)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
#-------------------------------------初始設定----------------------------------------------

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    if not cam.isOpened():
        print("Cannot open camera")
        exit()

    while program:
        ret, img = cam.read()
        img = cv2.flip(img, 1)   
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if not ret:
            print("Cannot receive frame")
            break
        
        results = hands.process(imgRGB)            #偵測手掌
        if results.multi_hand_landmarks and control==1:
            handnum = 0
            for hand,hand_landmarks in zip(results.multi_handedness,results.multi_hand_landmarks):
                
                hand_draw()

                handtype = hand.classification[0].index
                #handstype.append(handtype) 
                finger_points = []               #記錄手指節點座標的串列
                for i in hand_landmarks.landmark:#將 21 個節點換算成座標，記錄到 finger_points
                    x = i.x*540
                    y = i.y*310
                    finger_points.append((x,y))
                if finger_points:
                    finger_angle = hand_angle(finger_points)#計算手指角度，回傳長度為 5 的串列
                #    now = hand_pos(finger_angle)           #取得手勢所回傳的內容   

                if handtype == (mainhand+1)%2:#慣用手
                    handnum = handnum+1
                    pose_main = hand_pos(finger_angle)
                    poselist_main=list_updata(poselist_main,pose_main)
                    if np.max(poselist_main) == np.min(poselist_main):
                        pose_main_sure = np.max(poselist_main)
                        if pose_main_sure!=pv_main:
                            if pose_main_sure == 5:
                                mouse.release('left')
                            elif pose_main_sure == 1:
                                mouse.press('left')
                            elif pose_main_sure == 8:
                                mouse.double_click('left')
                            elif pose_main_sure == 4:
                                mouse.click('left')
                            pv_main=pose_main_sure
                    
                    xm8,ym8=hand_landmarks.landmark[8].x*camw,hand_landmarks.landmark[8].y*camh
                    xm2,ym2=hand_landmarks.landmark[2].x*camw,hand_landmarks.landmark[2].y*camh
                    
                    xm2d,ym2d=pos_change(hand_landmarks.landmark[2].x,hand_landmarks.landmark[2].y)
                    #"""
                    x_main,y_main = pos_change(hand_landmarks.landmark[8].x,hand_landmarks.landmark[8].y)
                    xm8d,ym8d=x_main,y_main
                    #-----抖動修正-----
                    xlist=list_updata(xlist,x_main)
                    ylist=list_updata(ylist,y_main)

                    xr=int((np.mean(xlist)*65536)/w)#轉換為絕對座標
                    yr=int((np.mean(ylist)*65536)/h)#轉換為絕對座標
                    #-----抖動修正-----
                    if pose_main_sure != 0 and pose_main_sure != 7:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE|win32con.MOUSEEVENTF_ABSOLUTE,xr, yr)#控制滑鼠游標
                    
                elif handtype == (mainhand)%2:#非慣用手
                    handnum = handnum+1
                    pose_sub = hand_pos(finger_angle)
                    poselist_sub=list_updata(poselist_sub,pose_sub)
                    if np.max(poselist_sub) == np.min(poselist_sub):
                        pose_sub_sure = np.max(poselist_sub)
                        if pose_sub_sure!=pv_sub:
                            if pose_sub_sure == 4:
                                mouse.click('right')
                            pv_sub=pose_sub_sure
                    
                    xs8,ys8=hand_landmarks.landmark[8].x*camw,hand_landmarks.landmark[8].y*camh
                    xs2,ys2=hand_landmarks.landmark[2].x*camw,hand_landmarks.landmark[2].y*camh
                    
                    xs2d,ys2d=pos_change(hand_landmarks.landmark[2].x,hand_landmarks.landmark[2].y)
                    #"""
                    x_sub,y_sub = pos_change(hand_landmarks.landmark[8].x,hand_landmarks.landmark[8].y)
                    xs8d,ys8d = x_sub,y_sub
                    #cv2.rectangle(img,(int((camw_h-w_h/sens)+x_sub-w_h),int((camh_h-h_h/sens)+y_sub-h_h)),(int((camw_h+w_h/sens)+x_sub-w_h),int((camh_h+h_h/sens)+y_sub-h_h)),(0,0,255),1)
                    """
                    #-----抖動修正-----
                    xlist=list_updata(xlist,xn)
                    ylist=list_updata(ylist,yn)
                    x_sub=int((np.mean(xlist)*65536)/w)#轉換為絕對座標
                    y_sub=int((np.mean(ylist)*65536)/h)#轉換為絕對座標
                    #-----抖動修正-----
                    """

                if handnum==2:
                    
                    if pose_sub_sure == 7 and pose_main_sure == 7:
                        
                        xs=int(min(xs8d,xs2d,xm8d,xm2d))
                        ys=int(min(ys8d,ys2d,ym8d,ym2d))
                        xm=int(max(xs8d,xs2d,xm8d,xm2d))
                        ym=int(max(ys8d,ys2d,ym8d,ym2d))

                        if xs<0:
                            xs=0
                        if ys<0:
                            ys=0
                        xsdlist=list_updata(xsdlist,xs)
                        ysdlist=list_updata(ysdlist,ys)
                        xs=int(np.mean(xsdlist))
                        ys=int(np.mean(ysdlist))
                        if not zoom:
                            img_desktop = ImageGrab.grab(bbox=(0, 0, w, h),include_layered_windows = 0)
                            frame_desktop = np.array(img_desktop)
                            RGB_frame_desktop=cv2.cvtColor(frame_desktop,cv2.COLOR_BGR2RGB)
                        crop_img = RGB_frame_desktop[ys+1:ym, xs+1:xm]
                        crop_img = cv2.resize(crop_img,(int((xm-xs)*1.5),int((ym-ys)*1.5)))
                        cv2.imshow('zoom in', crop_img)
                        zoom=1
                        if catch == 1:
                            cv2.imwrite('catch.jpg', crop_img)
                        #"""
                    elif pose_sub_sure == 8 and pose_main_sure == 7:
                        xs=int(min(xs8,xs2,xm8,xm2))
                        ys=int(min(ys8,ys2,ym8,ym2))
                        xm=int(max(xs8,xs2,xm8,xm2))
                        ym=int(max(ys8,ys2,ym8,ym2))

                        if xs<0:
                            xs=0
                        if ys<0:
                            ys=0
                        xslist=list_updata(xslist,xs)
                        yslist=list_updata(yslist,ys)
                        xs=int(np.mean(xslist))
                        ys=int(np.mean(yslist))
                        crop_img = img[ys+1:ym, xs+1:xm] 
                        crop_img = cv2.resize(crop_img,(int((xm-xs)*1.5),int((ym-ys)*1.5)))
                        cv2.rectangle(img,(xs,ys),(xm,ym),(0,0,255),1)
                        cv2.imshow('zoom in', crop_img)
                        zoom=1
                        if catch == 1:
                            cv2.imwrite('catch.jpg', crop_img)
                        
                    elif zoom:
                        cv2.destroyWindow('zoom in')
                        zoom = 0
                        
                    handnum = 0
                    
        if window:
            
            cv2.rectangle(img,(int(camw_h-w_h/sens),int(camh_h-h_h/sens)),(int(camw_h+w_h/sens),int(camh_h+h_h/sens)),(0,0,255),1)
            img = cv2.resize(img,(640,360))
            cv2.imshow('Seanut', img)

        if pose_main == 6:
            stop_wait_sec = stop_wait_sec + 1
        else:
            stop_wait_sec = 0
        if cv2.waitKey(1) == ord('q') or stop_wait_sec == 30:
            break 

program = 0
cam.release()
cv2.destroyAllWindows()
