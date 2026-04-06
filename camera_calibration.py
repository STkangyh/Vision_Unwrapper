import cv2 as cv
import numpy as np

def select_img_from_video(video_path, board_pattern, select_all=False, wait_msec=10):
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        print("Error opening video stream or file")
        return None
    img_selected = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        complete, _ = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_selected.append(frame)
            if not select_all:
                break
        cv.imshow('Video', frame)
        if cv.waitKey(wait_msec) & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()

    return img_selected

def select_img_from_camera(video_path, board_pattern, select_all=False, wait_msec=10):
    cap = cv.VideoCapture(0)  # Open the default camera
    img_selected = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        print("Checking for chessboard corners...")
        complete, _ = cv.findChessboardCorners(gray, board_pattern)
        print(f"Chessboard corners found: {complete}")
        if complete:
            img_selected.append(frame)
            print("Chessboard corners found and image selected.")
            if not select_all:
                print("Selected one image, exiting selection loop.")
                break
        cv.imshow('Camera', frame)
        if cv.waitKey(wait_msec) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    return img_selected

def calib_camera_from_chessboard(img, board_pattern, board_cellsize, K=None, dist_coeffs=None, calib_flags=None):
    # Find 2D corner points in the image
    img_points = []
    for img in img:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0, "No chessboard corners found in any image."

    # Prepare 3D object points based on the chessboard pattern and cell size
    obj_points = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_points, dtype=np.float32) * board_cellsize] * len(img_points)

    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeffs, flags=calib_flags)

def save_calib_result(filename, mtx, dist):
    np.savez(filename, mtx=mtx, dist=dist)
    print(f"Calibration results saved to {filename}")

if __name__ == "__main__":
    # Parameters
    video_path = 'calib_video.avi'  # Path to the calibration video
    board_pattern = (10, 7)           # Chessboard pattern (columns, rows)
    board_cellsize = 25.0            # Size of each chessboard cell in mm

    # Select images from the video for calibration
    img_selected = select_img_from_video(video_path, board_pattern, select_all=True)

    if not img_selected:
        print("No valid images selected from video. Trying camera...")
        img_selected = select_img_from_camera(video_path, board_pattern, select_all=True)

    if img_selected:
        # Calibrate the camera using the selected images
        ret, mtx, dist, rvecs, tvecs = calib_camera_from_chessboard(img_selected, board_pattern, board_cellsize)

        if ret:
            print("Camera calibration successful.")
            print("Camera matrix:\n", mtx)
            print("Distortion coefficients:\n", dist)

            # Save the calibration results
            save_calib_result('calib_result.npz', mtx, dist)
        else:
            print("Camera calibration failed.")
    else:
        print("No valid images selected for calibration.")
