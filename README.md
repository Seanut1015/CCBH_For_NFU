# CCBH_For_NFU
## 簡介
實現影視作品中常見的操作模式，利用手部辨識配合語音操作  

## 注意
此程式目前還在測試階段  
部分功能透過win32API實現  
請不要在遊戲時使用，可能被判斷為腳本  

## 使用說明
### 基本調整與注意事項
**請務必將config文件與主程式放置於同一目錄下**

首次開啟時，調整視訊鏡頭畫面至可以拍攝到雙手的位置  
測試左右手移動範圍，並調整靈敏度  
偵測點為慣用手食指指腹  
### 目前手勢
###### 慣用手：
    張開：	無動作
    手比1：	按住左鍵
    手比7：	雙擊左鍵
    手比4：	單擊左鍵
    手比6：	結束程式
###### 非慣用手：
    張開：	無動作
    手比4：	單擊右鍵
### 操作細項調整
透過修改config文件變更設定

###### 慣用手(mainhand)
	預設值：0	#右手：0、左手：1
###### 抖動修正(unshake)
	預設值：2	#數值越高延遲越重，請使用正整數，建議不要超過4
###### 螢幕解析度(display)
	預設值：w:1920 h:1080	#請先調整為當前螢幕解析度
	#如果操作卡頓，請將數值等比例調低 *可能造成精度降低*
###### 攝影機解析度(cam)
	預設值：w:1280 h:720	#如果操作卡頓，請將數值等比例調低 *可能造成精度降低*
###### 靈敏度(sens)
	攝影畫面(windows)	#預設值：1.875	#調整操控靈敏度
###### 攝影畫面(windows)
	預設值：1	#關閉：0、開啟：1
## 模組
	pip3 install opencv-python
	pip3 install opencv-contrib-python
	pip3 install mediapipe
	pip3 install numpy
	pip3 install configparser
	pip3 install pywin32
	pip3 install mouse
	pip3 install pyaudio
	pip3 install pyttsx3
	pip3 install SpeechRecognition
	pip3 install pyqt5
