import cv2
import numpy as np
import threading
from datetime import datetime

# ─── 터미널 입력 스레드 (macOS에서 cv2.waitKey 키 인식 불가 문제 우회) ─────────
stop_flag = threading.Event()

def wait_for_quit():
    """터미널에서 Enter 키를 누르면 종료 신호를 보낸다."""
    input()          # 터미널에서 Enter 대기
    stop_flag.set()

input_thread = threading.Thread(target=wait_for_quit, daemon=True)
input_thread.start()

# ─── 캘리브레이션 파라미터 로드 ────────────────────────────────────────────────
try:
    with np.load('calib_result.npz') as data:
        mtx  = data['mtx']
        dist = data['dist']
except FileNotFoundError:
    print("캘리브레이션 결과 파일이 없습니다. 먼저 camera_calibration.py를 실행하세요.")
    exit()

# ─── 카메라 열기 ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] 카메라를 열 수 없습니다.")
    exit()

print("[INFO] 실행 중...  종료하려면 터미널에서 Enter 를 누르세요.")

last_frame = None
last_dst   = None

while not stop_flag.is_set():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # 마지막 프레임 보관
    last_frame = frame.copy()
    last_dst   = dst.copy()

    # 라벨 추가 후 가로 합성
    label_orig = frame.copy()
    label_dst  = dst.copy()
    cv2.putText(label_orig, 'Original (Distorted)',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(label_dst,  'Corrected (Undistorted)',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    combined = np.hstack((label_orig, label_dst))

    cv2.imshow('Original | Corrected  (터미널에서 Enter 로 종료)', combined)
    cv2.waitKey(1)   # 창 렌더링 유지용 (키 입력은 터미널로 처리)

# ─── 종료 시 마지막 프레임 저장 ────────────────────────────────────────────────
if last_frame is not None and last_dst is not None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    orig_path = f'original_{timestamp}.png'
    corr_path = f'corrected_{timestamp}.png'
    cv2.imwrite(orig_path, last_frame)
    cv2.imwrite(corr_path, last_dst)
    print(f"[저장 완료] 원본   → {orig_path}")
    print(f"[저장 완료] 보정본 → {corr_path}")
else:
    print("[경고] 저장할 프레임이 없습니다.")

cap.release()
cv2.destroyAllWindows()