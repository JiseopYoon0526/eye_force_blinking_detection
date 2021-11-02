import dlib
import skvideo.io
import cv2

## face detector와 landmark predictor 정의
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

## 비디오 읽어오기
cap = skvideo.io.vreader('2.mp4')
# cap = cv2.VideoCapture(0)
a=0
## 각 frame마다 얼굴 찾고, landmark 찍기

for frame in cap:    
# while(True):
    #ret, frame = cap.read() 
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    r = 200. / img.shape[1]
    dim = (200, int(img.shape[0] * r))    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    rects = detector(resized, 1)
    for i, rect in enumerate(rects):
        # 인식된 얼굴 사각형 설정
        l = rect.left()
        t = rect.top()
        b = rect.bottom()
        r = rect.right()
        shape = predictor(resized, rect)
        # 눈썹 및 눈 윗 부분 추출
        for j in range(37,40):
            x, y = shape.part(j).x, shape.part(j).y
            cv2.circle(resized, (x, y), 1, (0, 0, 255), -1)
        for j in range(18,27):
            x, y = shape.part(j).x, shape.part(j).y
            cv2.circle(resized, (x, y), 1, (0, 0, 255), -1)
        for j in range(37,40):
            x, y = shape.part(j).x, shape.part(j).y
            cv2.circle(resized, (x, y), 1, (0, 0, 255), -1)
        for j in range(43,46):
            x, y = shape.part(j).x, shape.part(j).y
            cv2.circle(resized, (x, y), 1, (0, 0, 255), -1)
        # 얼굴에 사각형 그리기
        cv2.rectangle(resized, (l, t), (r, b), (0, 255, 0), 2)
        if(a%30==0&abs(shape.part(24).y-shape.part(44).y)<8&abs(shape.part(19).y-shape.part(37).y)<8):
            print("왼쪽 세게 깜빡였음 오른쪽 세게 깜빡였음",a/30,"초")
        elif(a%30==0&abs(shape.part(24).y-shape.part(44).y)>8&abs(shape.part(19).y-shape.part(37).y)<8):
            print("왼쪽 세게 깜빡임 오른쪽 약하게 깜빡임",a/30,"초")
        elif(a%30==0&abs(shape.part(24).y-shape.part(44).y)<8&abs(shape.part(19).y-shape.part(37).y)>8):
            print("왼쪽 약하게 깜빡임 오른쪽 세게 깜빡임",a/30,"초")
        elif(a%30==0):
            print("왼쪽 약하게 깜빡임 오른쪽 약하게 깜빡임",a/30,"초")

        a+=1
        #영상 추출
        cv2.imshow('frame', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
