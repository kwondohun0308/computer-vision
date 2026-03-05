import cv2 as cv
import sys

brush_size = 5 # 초기값 5로 설정
drawing = False # 계속 그리고 있는지 여부를 확인할 변수

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