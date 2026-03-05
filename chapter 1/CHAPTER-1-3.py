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

img_original = img.copy()  # 원본 이미지 백업

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
