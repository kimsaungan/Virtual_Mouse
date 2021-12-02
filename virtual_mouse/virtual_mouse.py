import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100     # Frame Reduction 
smoothening = 7
#########################

pTime=0
plocX,plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)  
cap.set(3,wCam)    #  영상출력 콘솔창 크기 설정
cap.set(4,hCam)    #  영상출력 콘솔창 크기 설정

detector=htm.handDetector(maxHands=1) 
wScr,hScr = autopy.screen.size()
#print(wScr,hScr)

while True:
    #1. 손가락 랜드마크 찾기
    success,img = cap.read()
    img=detector.findHands(img)
    lmList,bbox=detector.findPosition(img)   # 손 인식과 손 인식 영역에 직사각형으로 표시 
    
    #2. 검지, 중지 끝의 인덱스 가져오기 
    if len(lmList)!=0:
        
        x1,y1=lmList[8][1:]      # 검지 끝 인덱스 검출
        x2,y2=lmList[12][1:]     # 중지 끝 인덱스 검출
        print(x1,y1,x2,y2)
        
        #3. 손가락이 펴져있는지 여부 판별
        fingers= detector.fingersUp()    # ex) [0,1,1,0,0] = 검지, 중지 만 펴져있는 상태
        print(fingers)
        cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
        
        
        #4. 검지만 펴져있을때 마우스 무빙 모드  작동
        if fingers[1]==1 and fingers[2]==0:
            #5. 좌표 변환 (영상 속 손가락 인식 화면의 좌표와 컴퓨터 화면상 좌표를 비율을 맞추기 위해 )
            
            x3= np.interp(x1,(frameR,wCam-frameR),(0,wScr))      #np.interp()을 통해 선형보간법 적용
            y3= np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            
            #6. Smoothing Value 값 설정 마우스를 구동할때 진동을 감쇠시키기 위한 값 설정
            clocX = plocX+(x3-plocX)/smoothening
            clocY = plocY+(y3-plocY)/smoothening
            
            
            #7. 마우스 움직이기
            autopy.mouse.move(wScr-clocX, clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)   # 검지 끝에 큰 원을 출력
            plocX, plocY = clocX, clocY
            
            
        #8. 검지, 중지를 펴면 = 클릭 모드 구현
        if fingers[1]==1 and fingers[2]==1:   #  fingers[1]== 1 : 검지 끝을 편 상태, fingers[2]== 중지 끝을 편 상태
            #9. 검지,중지 사이의 간격 측정
            length, img, lineInfo= detector.findDistance(8, 12, img)
            print(length)
            
            #10. 검지,중지 사이 간격을 통해 클릭기능  활성화
            if length<40:                                                           # 검지, 중지 사이 간격이 40 이하 이면
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)   # 중간 원을 노란색으로 출력
                autopy.mouse.click()
            
    
    #11. 화면 비율
    cTime = time.time()
    fps=1/(cTime-pTime)    # 초당 프레임 수(frames per second,)는 1초 동안 보여주는 화면의 수를 가리킴
    pTime=cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0), 3)
    
    #12. 출력 
    cv2.imshow("image",img)
    cv2.waitKey(1)
    
    
    