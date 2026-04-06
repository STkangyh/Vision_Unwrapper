import cv2
import numpy as np

# 앞서 저장한 캘리브레이션 파라미터 불러오기
try:
    with np.load('calib_result.npz') as data:
        mtx = data['mtx']
        dist = data['dist']
except FileNotFoundError:
    print("캘리브레이션 결과 파일이 없습니다. 먼저 camera_calibration.py를 실행하세요.")
    exit()

# 왜곡을 보정할 카메라 또는 동영상 불러오기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    # 새로운 카메라 매트릭스 계산 (자유도 조절 가능: 1이면 모든 픽셀 보존, 0이면 유효 픽셀만 크롭)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 렌즈 왜곡 보정 수행
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # 원본(왜곡된) 영상과 보정된 영상 비교
    cv2.imshow('Original (Distorted)', frame)
    cv2.imshow('Corrected (Undistorted)', dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()