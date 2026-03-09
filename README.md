# chapter 1
## 과제1 설명 및 요구사항 (이미지 불러오기 및 그레이스케일 변환)
 - 원본 이미지와 그레이스케일로 변환된 이미지를 나란히 표시
 - cv.imread()를 사용하여 이미지를 로드
 - cv.cvtColor() 함수를 이용해 이미지를 그레이스케일로 변환
 - np.hstack() 함수를 이용해 원본 이미지와 그레이스케일 이미지를 가로로 연결하여 출력
 - cv.imshow()와 cv.waitKey()를 사용해 결과를 화면에 표시하고, 아무 키나 누르면 창이 닫히도록 할 것
<img width="1436" height="506" alt="1번 문제" src="https://github.com/user-attachments/assets/60f2b5d4-2218-4aad-8672-a2dcfbe94580" />

과제 1번 전체 코드

```python
import pandas as pd
import numpy as np
import cv2 as cv
import sys

img = cv.imread('soccer.jpg')
if img is None:
    print('Image load failed!')
    sys.exit()

img_small = cv.resize(img, (0, 0), fx=0.5, fy=0.5) #이미지 크기를 반으로 감소
#그대로 이어붙이면 너무 커서 사진 사이즈를 1/4로 줄임
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #이미지를 흑백으로 변환
gray_small = cv.resize(gray, (0, 0), fx=0.5, fy=0.5) #흑백으로 변환한 이미지 크기를 반으로 감소
#그대로 이어붙이면 너무 커서 사진 사이즈를 1/4로 줄임

# 원본 이미지는 BGR 3채널인데, 흑백 이미지는 1채널이므로 흑백 이미지의 픽셀값을 복사해서 32채널로 만들어 원본과 구조를 맞춰줬음
gray_3ch = cv.cvtColor(gray_small, cv.COLOR_GRAY2BGR)
combined = np.hstack((img_small, gray_3ch))

cv.imshow('combined', combined)
cv.waitKey()
cv.destroyAllWindows()

```

## 과제2 설명 및 요구사항 (페인팅 붓 크기 조절 기능 추가)
 - 마우스 입력으로 이미지 위에 붓질
 - 키보드 입력을 이용해 붓의 크기를 조절하는 기능 추가
 - 초기 붓 크기는 5를 사용
 - '+' 입력 시 붓 크기 1 증가, '-' 입력 시 붓 크기 1 감소
 - 붓 크기는 최소 1, 최대 15로 제한
 - 좌클릭=파란색, 우클릭=빨간색, 드래그로 연속 그리기
 - q 키를 누르면 영상 창이 종료
<img width="1436" height="980" alt="2번 문제" src="https://github.com/user-attachments/assets/6779d771-ea3a-4e8f-8eb6-ddc4062a1e2f" />

과제 2번 전체 코드

```python
import cv2 as cv
import sys

brush_size = 5 # 초기값 5로 설정
drawing = False # 클릭한 상태에서 이동할때 그려져야 하므로 스위치 역할인 drawing을 변수로 사용함

def draw_on_image(event, x, y, flags, param):
    global drawing, brush_size
    
    # 좌클릭은 파란색
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        cv.circle(img, (x, y), brush_size, (255, 0, 0), -1)
    
    # 우클릭은 빨간색
    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = True
        cv.circle(img, (x, y), brush_size, (0, 0, 255), -1)
    
    # 마우스가 이동중에 좌클릭이나 우클릭이 유지되고 있다면 계속 그리기
    elif event == cv.EVENT_MOUSEMOVE and drawing:
        if flags == cv.EVENT_FLAG_LBUTTON:
            cv.circle(img, (x, y), brush_size, (255, 0, 0), -1)
        elif flags == cv.EVENT_FLAG_RBUTTON:
            cv.circle(img, (x, y), brush_size, (0, 0, 255), -1)
    
    # 마우스 버튼 뗄 때 drawing 을 False로 설정
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        drawing = False

# 이미지 로드
img = cv.imread('soccer.jpg')
if img is None:
    print('soccer.jpg를 찾을 수 없습니다!')
    sys.exit()

# 윈도우 생성 및 마우스 콜백 등록
cv.namedWindow('image')
cv.setMouseCallback('image', draw_on_image)

print('\n=== 그림 그리기 프로그램 ===')
print('좌클릭: 파란색 | 우클릭: 빨간색')
print('+ : 붓 크기 증가 | - : 붓 크기 감소')
print('q : 종료\n')

# 메인 루프
while True:
    cv.imshow('image', img)
    key = cv.waitKey(1) & 0xFF #& 0xFF는 64비트 시스템에서 키 입력을 올바르게 처리하기 위해 사용 #이게 없다면 64비트 시스템에서 키 입력이 제대로 인식되지 않을 수 있음
    
    if key == ord('q'):# q 키를 누르면 프로그램 종료
        print('프로그램을 종료합니다.')
        break
    
    # 붓 크기 조절
    # + 또는 = 키를 누르면 붓 크기 증가, - 키를 누르면 붓 크기 감소
    elif key == ord('+') or key == ord('='):
        brush_size = min(brush_size + 1, 15)# 붓 크기를 최대 15로 제한
        print(f'현재 붓 크기: {brush_size}')
    
    elif key == ord('-'):
        brush_size = max(brush_size - 1, 1)# 붓 크기를 최소 1로 제한
        print(f'현재 붓 크기: {brush_size}')

cv.destroyAllWindows()# 프로그램 종료 후 모든 창 닫기
```


## 과제3 설명 및 요구사항 (마우스로 영역 선택 및 ROI 추출)
 - 이미지를 불러오고 사용자가 마우스로 클릭하고 드래그하여 관심영역을 선택
 - 선택한 영역만 따로 저장하거나 표시
 - 이미지를 불러오고 화면에 출력
 - cv.setMouseCallback()을 사용하여 마우스 이벤트를 처리
 - 사용자가 클릭한 시작점에서 드래그하여 사각형을 그리며 영역을 선택
 - 마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력
 - r 키를 누르면 영역 선택을 리셋하고 처음부터 다시 선택
 - s 키를 누르면 선택한 영역을 이미지 파일로 저장

<img width="1436" height="980" alt="3번 문제" src="https://github.com/user-attachments/assets/2939b9d5-d71c-41e5-a48b-fa087342d110" />

과제 3번 전체 코드

```python
import cv2 as cv
import sys

start_point = None  # 시작 좌표
end_point = None    # 끝 좌표
drawing = False     # 드래그 중인지 확인
img_copy = None     # 원본 이미지 복사본
selected_roi = None # 선택된 영역 저장

def select_roi(event, x, y, flags, param): # 마우스 이벤트 콜백 함수
    global start_point, end_point, drawing, img, img_copy, img_original
    
    # 좌클릭을 하면 시작 좌표 저장
    if event == cv.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        end_point = (x, y)
        drawing = True
        img_copy = img_original.copy()
    
    # drawing이 True인 상태에서 마우스가 이동하면 사각형 그리기
    elif event == cv.EVENT_MOUSEMOVE and drawing:
        img[:] = img_copy[:]  # 이전 프레임 복원 # 이게 없다면 드래그할 때마다 사각형이 겹쳐서 그려지는 문제가 발생
        end_point = (x, y)
        cv.rectangle(img, start_point, end_point, (0, 255, 0), 2)

    # 마우스 버튼을 놓으면 영역 잘라내기
    elif event == cv.EVENT_LBUTTONUP:
        global selected_roi
        drawing = False
        end_point = (x, y)
        
        # 시작점과 끝점으로 사각형 좌표 계산
        x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
        
        # 영역이 있을 때만 잘라내기
        if x2 - x1 > 0 and y2 - y1 > 0:
            selected_roi = img_copy[y1:y2, x1:x2]  # 선택 영역 저장
            cv.imshow('Selected ROI', selected_roi)  # 별도 창에 출력
        
        img[:] = img_copy[:]  # 원본 이미지 복원 # 드래그가 끝난 후에도 사각형이 남아있지 않도록 원본 이미지로 복원

# 이미지 로드
img = cv.imread('soccer.jpg')
if img is None:
    print('soccer.jpg를 찾을 수 없습니다!')
    sys.exit()

img_original = img.copy()  # 원본 이미지 백업 r키를 눌렀을때 초기화 되어야 하므로

# 윈도우 생성 및 마우스 콜백 등록
cv.namedWindow('image')
cv.setMouseCallback('image', select_roi)

# 메인 루프 # 영역 선택, 리셋, 저장 기능을 위한 키 입력 처리
while True:
    cv.imshow('image', img)
    key = cv.waitKey(1) & 0xFF
    
    # q 키로 프로그램 종료
    if key == ord('q'):
        print('프로그램을 종료합니다.')
        break
    
    # r키로 리셋
    elif key == ord('r'):
        img[:] = img_original[:]
        try:
            cv.destroyWindow('Selected ROI')
        except:
            pass
        selected_roi = None
        print('영역 선택이 리셋되었습니다.')
    
    # s키로 선택 영역 저장
    elif key == ord('s'):
        if selected_roi is not None:
            cv.imwrite('selected_region.jpg', selected_roi)
            print('선택 영역이 selected_region.jpg로 저장되었습니다.')
        else:
            print('저장할 영역이 없습니다. 먼저 영역을 선택하세요.')

cv.destroyAllWindows()
```
