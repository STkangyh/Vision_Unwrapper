import cv2 as cv
import numpy as np

# ─── 1. 캘리브레이션 파라미터 로드 ─────────────────────────────────────────────
def load_calib_result(filename='calib_result.npz'):
    """저장된 카메라 캘리브레이션 결과를 불러온다."""
    try:
        with np.load(filename) as data:
            mtx  = data['mtx']
            dist = data['dist']
        print(f"[INFO] Calibration result loaded from '{filename}'")
        return mtx, dist
    except FileNotFoundError:
        print(f"[ERROR] '{filename}' not found. Run camera_calibration.py first.")
        exit()


# ─── 2. 체스보드 코너 검출 및 solvePnP ────────────────────────────────────────
def estimate_pose(frame, board_pattern, board_cellsize, K, dist):
    """
    단일 프레임에서 체스보드를 찾아 카메라 포즈(R, t)를 추정한다.

    Returns
    -------
    success : bool
    rvec    : (3,1) 회전 벡터 (Rodrigues)
    tvec    : (3,1) 이동 벡터 (단위: board_cellsize 와 동일)
    corners : 검출된 코너 픽셀 좌표
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    complete, corners = cv.findChessboardCorners(gray, board_pattern)

    if not complete:
        return False, None, None, None

    # 코너 정밀도 향상 (서브픽셀 정제)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 3D 물체 좌표 (체스보드 평면, Z=0)
    obj_pts = np.array(
        [[c * board_cellsize, r * board_cellsize, 0]
         for r in range(board_pattern[1])
         for c in range(board_pattern[0])],
        dtype=np.float32
    )

    success, rvec, tvec = cv.solvePnP(obj_pts, corners, K, dist)
    return success, rvec, tvec, corners


# ─── 3. 시각화 헬퍼 ────────────────────────────────────────────────────────────
def draw_axes(frame, K, dist, rvec, tvec, axis_length):
    """
    카메라 좌표계의 X(빨강), Y(초록), Z(파랑) 축을 이미지에 그린다.
    axis_length: 축 길이 (board_cellsize 단위)
    """
    origin = np.float32([[0, 0, 0]])
    axes_3d = np.float32([
        [axis_length, 0, 0],   # X
        [0, axis_length, 0],   # Y
        [0, 0, -axis_length],  # Z (카메라 방향)
    ])

    img_origin, _ = cv.projectPoints(origin, rvec, tvec, K, dist)
    img_axes,   _ = cv.projectPoints(axes_3d, rvec, tvec, K, dist)

    o  = tuple(img_origin[0].ravel().astype(int))
    px = tuple(img_axes[0].ravel().astype(int))
    py = tuple(img_axes[1].ravel().astype(int))
    pz = tuple(img_axes[2].ravel().astype(int))

    cv.arrowedLine(frame, o, px, (0, 0, 255), 3, tipLength=0.2)   # X: 빨강
    cv.arrowedLine(frame, o, py, (0, 255, 0), 3, tipLength=0.2)   # Y: 초록
    cv.arrowedLine(frame, o, pz, (255, 0, 0), 3, tipLength=0.2)   # Z: 파랑


def draw_cube(frame, K, dist, rvec, tvec, cube_size):
    """
    체스보드 원점 위에 정육면체를 그린다.
    cube_size: 한 변의 길이 (board_cellsize 단위)
    """
    s = cube_size
    # 바닥면(Z=0), 윗면(Z=-s)
    cube_3d = np.float32([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],   # 바닥
        [0, 0,-s], [s, 0,-s], [s, s,-s], [0, s,-s],   # 윗면
    ])

    pts, _ = cv.projectPoints(cube_3d, rvec, tvec, K, dist)
    pts    = pts.reshape(-1, 2).astype(int)

    # 바닥면
    for i in range(4):
        cv.line(frame, tuple(pts[i]),   tuple(pts[(i+1) % 4]), (0, 255, 255), 2)
    # 윗면
    for i in range(4):
        cv.line(frame, tuple(pts[i+4]), tuple(pts[(i+1) % 4 + 4]), (0, 165, 255), 2)
    # 기둥
    for i in range(4):
        cv.line(frame, tuple(pts[i]),   tuple(pts[i+4]),          (255, 255, 0), 2)


def overlay_pose_info(frame, rvec, tvec):
    """회전·이동 벡터 수치를 프레임 좌상단에 출력한다."""
    R, _ = cv.Rodrigues(rvec)
    # 오일러 각 (ZYX 순서)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll  = np.degrees(np.arctan2( R[2,1],  R[2,2]))
        pitch = np.degrees(np.arctan2(-R[2,0],  sy))
        yaw   = np.degrees(np.arctan2( R[1,0],  R[0,0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1,2],  R[1,1]))
        pitch = np.degrees(np.arctan2(-R[2,0],  sy))
        yaw   = 0.0

    tx, ty, tz = tvec.ravel()
    lines = [
        f"Roll : {roll:7.2f} deg",
        f"Pitch: {pitch:7.2f} deg",
        f"Yaw  : {yaw:7.2f} deg",
        f"Tx   : {tx:7.2f}",
        f"Ty   : {ty:7.2f}",
        f"Tz   : {tz:7.2f}",
    ]
    for i, line in enumerate(lines):
        cv.putText(frame, line, (10, 25 + i * 22),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# ─── 4. 메인 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ── 파라미터 설정 ──────────────────────────────────────────────────────────
    BOARD_PATTERN  = (10, 7)   # 체스보드 내부 코너 수 (열, 행)
    BOARD_CELLSIZE = 25.0      # 체스보드 한 칸 크기 (mm 또는 임의 단위)
    CALIB_FILE     = 'calib_result.npz'
    VIDEO_SOURCE   = 0         # 0: 웹캠, 또는 영상 파일 경로 문자열

    # ── 초기화 ────────────────────────────────────────────────────────────────
    K, dist = load_calib_result(CALIB_FILE)
    cap     = cv.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        exit()

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        # 렌즈 왜곡 보정
        h, w = frame.shape[:2]
        new_K, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undist      = cv.undistort(frame, K, dist, None, new_K)

        # 포즈 추정
        success, rvec, tvec, corners = estimate_pose(
            undist, BOARD_PATTERN, BOARD_CELLSIZE, new_K, None
        )

        if success:
            # 코너 그리기
            cv.drawChessboardCorners(undist, BOARD_PATTERN, corners, success)

            # 축 및 큐브 그리기
            draw_axes(undist, new_K, None, rvec, tvec,
                      axis_length=BOARD_CELLSIZE * 3)
            draw_cube(undist, new_K, None, rvec, tvec,
                      cube_size=BOARD_CELLSIZE * 2)

            # 수치 오버레이
            overlay_pose_info(undist, rvec, tvec)

            cv.putText(undist, "Pose: DETECTED", (10, undist.shape[0] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv.putText(undist, "Pose: NOT DETECTED", (10, undist.shape[0] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv.imshow('Pose Estimation', undist)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
