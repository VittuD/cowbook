import os, json, math

import numpy as np
from numpy.linalg import inv
import cv2 as cv

from . import points_data as pts_data

CAMERA_WIDTH = 2688
CAMERA_HEIGHT = 1520

UNDISTORTION_COEFFICENT = 0

def get_calibrated_camera_model(
    calibration_model_path="legacy/calibration_matrix.json",
):
    """
    Check if in given path a camera model is already present, if return it, otherwise tell user to calibrate the camera first

    Keyword arguments:
    calibration_model_path -- path where model is stored

    Return camera intrinsic matrix and distortion coefficients.
    """

    if not os.path.isfile(calibration_model_path):
        print("Camera model not found")
        print("Calibrate the camera running the 'camera_calibrator' script")

    with open(calibration_model_path) as f:
        data = json.load(f)

        return np.asarray(data["camera_matrix"]), np.asarray(data["dist_coeff"])


def show_img(window_name, img):
    """
    Show an image with a resizable window
    This is useful given that OpenCV by default create a shitty window

    Keyword arguments:
    window_name -- window name
    img -- image matrix to show
    """
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1920, 1080)
    cv.imshow(window_name, img)

def save_frame_image(projected_points, frame_num, output_path,
                     barn_image_path="legacy/barn.png", barn_image=None):
    """
    Save processed frame with projected points to an image file.

    Args:
        projected_points (list): List of (x, y[, z]) points projected on the ground plane.
        frame_num (int): Frame number (used only for logging/text overlays if desired).
        output_path (str): Output filename (extension controls format, e.g. .png or .jpg).
        barn_image_path (str): Fallback path to load the barn background if `barn_image` is None.
        barn_image (np.ndarray | None): Optional preloaded background image. If provided,
                                        it will be copied and reused to avoid disk I/O per frame.
    """
    # Use preloaded background if given; otherwise load from disk, else make a blank canvas
    if barn_image is not None:
        img = barn_image.copy()
    elif os.path.exists(barn_image_path):
        img = cv.imread(barn_image_path)
    else:
        print(f"Warning: {barn_image_path} not found. Using blank background.")
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Draw projected points onto the background
    barn_points, drawn = points_to_barn(projected_points, img)

    # Optionally overlay frame index (comment out if not wanted)
    cv.putText(drawn, f"Frame: {frame_num}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    # Write the composed frame
    cv.imwrite(output_path, drawn)
    # print(f"Saved frame {frame_num} to {output_path}")

def show_taken_points(camera_nr):
    """
    Show the image with the plotted manually taken points

    Keyword arguments:
    camera_nr -- camera number to consider
    """
    img = cv.imread(f"assets/original_images/ch{camera_nr}.png")

    points = []
    if camera_nr == 1:
        points = pts_data.ch1_img
    elif camera_nr == 4:
        points = pts_data.ch4_img
    elif camera_nr == 8:
        points = pts_data.ch8_img
    elif camera_nr == 6:
        points = pts_data.ch6_img

    for p in points:
        cv.circle(
            img,
            tuple(p.astype(np.int32)),
            3,
            (0, 0, 255),
            10,
        )

    cv.namedWindow(f"points_ch{camera_nr}", cv.WINDOW_NORMAL)
    cv.resizeWindow(f"points_ch{camera_nr}", 1920, 1080)
    cv.imshow(f"points_ch{camera_nr}", img)


def get_rotation_matrix(angle):
    # Get camera center coordinates
    (cX, cY) = (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2)

    # Rotate our image by x degrees around the center of the image
    return cv.getRotationMatrix2D((cX, cY), -angle, 1.0)


def rotate_points(points, angle):
    """
    Rotate a given set of points around the center of camera

    Keyword arguments:
    points -- points to rotate
    angle -- angle expressed in degree

    Return the rotated points
    """

    # First get the rotation matrix based on the given angle, then transform the points
    M = get_rotation_matrix(angle)

    return cv.transform(points, M)


def rotate_image(img, angle):
    """
    Method to rotate a given image by x degree around its center

    Keyword arguments:
    img -- image matrix to rotate
    angle -- angle expressed in degrees

    Return the rotated image
    """

    (h, w) = img.shape[:2]

    # First get the rotation matrix and then rotate the image
    M = get_rotation_matrix(angle)

    return cv.warpAffine(img, M, (w, h))


def undistort_image(img, mtx, dist):
    """
    Undistort a given image with an input calibrated model

    Keyword arguments:
    img  -- image matrix to undistort
    mtx  -- camera intrinsic matrix
    dist -- camera distortion coefficients

    Return an undistorted image matrix
    """

    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), UNDISTORTION_COEFFICENT, (w, h)
    )
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)  # pyright: ignore
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]

    return dst


def undistort_points(camera_nr, mtx, dist):
    """
    Undistort a given set of points

    Keyword arguments:
    camera_nr -- camera where to take points from
    mtx       -- camera intrinsic matrix
    dist      -- camera distortion coefficients

    Return an undistorted set of points
    """

    points = []

    if camera_nr == 1:
        points = pts_data.ch1_img
    elif camera_nr == 4:
        points = pts_data.ch4_img
        points = rotate_points(np.array([points]), -1.5)
    elif camera_nr == 6:
        points = pts_data.ch6_img
    elif camera_nr == 8:
        points = pts_data.ch8_img
        points = rotate_points(np.array([points]), -1)

    h, w = CAMERA_HEIGHT, CAMERA_WIDTH
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), UNDISTORTION_COEFFICENT, (w, h)
    )
    # Use undistortPointsIter instead of undistortPoints for more accuracy
    undistorted_points = cv.undistortPointsIter(
        points,
        mtx,
        dist,
        None,
        newcameramtx,
        (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 40, 0.03),
    )

    return undistorted_points.reshape(undistorted_points.shape[0], 2).astype(int)

def undistort_points_given(points, mtx, dist):
    """
    Undistort a given set of points

    Keyword arguments:
    points    -- points to be undistorted
    mtx       -- camera intrinsic matrix
    dist      -- camera distortion coefficients

    Return an undistorted set of points
    """
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    h, w = CAMERA_HEIGHT, CAMERA_WIDTH
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), UNDISTORTION_COEFFICENT, (w, h)
    )
    # Use undistortPointsIter instead of undistortPoints for more accuracy
    undistorted_points = cv.undistortPointsIter(
        points,
        mtx,
        dist,
        None,
        newcameramtx,
        (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 40, 0.03),
    )

    # Keep float precision for better PnP/projection accuracy
    return undistorted_points.reshape(undistorted_points.shape[0], 2)

def draw_grid(img):
    """
    Print a grid with red lines and green ones for the center

    Keyword arguments:
    img -- image matrix to print on

    Return an image with a grid on
    """

    for i in range(1, 15):
        cv.line(img, (0, i * 100), (img.shape[1], i * 100), (150, 150, 150), 5)
    cv.line(
        img,
        (0, int(img.shape[0] / 2)),
        (img.shape[1], int(img.shape[0] / 2)),
        (0, 255, 0),
        5,
    )
    for i in range(1, 27):
        cv.line(img, (i * 100, 0), (i * 100, img.shape[0]), (150, 150, 150), 5)
    cv.line(
        img,
        (int(img.shape[1] / 2), 0),
        (int(img.shape[1] / 2), img.shape[0]),
        (0, 255, 0),
        5,
    )

    return img


def get_bird_view_perspective_transform_matrix(camera_nr, show=False, img=None):
    """
    Compute the perspective transform matrix based on a trapezium

    Keyword arguments:
    camera_nr -- camera number of which matrix is calculated

    Return the perspective transform matrix and the new width and height
    """

    if show:
        assert img is not None, "No image to show was given"

    pt_A, pt_B, pt_C, pt_D = [0, 0], [0, 0], [0, 0], [0, 0]

    # Points are taken anti clockwise starting from top left corner
    match camera_nr:
        case 1:
            pt_A = [960, 290]
            pt_B = [0, 1370]
            pt_C = [2600, 1370]
            pt_D = [1670, 290]
        case 4:
            pt_A = [965, 370]
            pt_B = [0, 1370]
            pt_C = [2687, 1370]
            pt_D = [1694, 370]
        case 6:
            pt_A = [1000, 335]
            pt_B = [0, 1370]
            pt_C = [2687, 1370]
            pt_D = [1694, 335]
        case 8:
            pt_A = [1045, 365]
            pt_B = [0, 1370]
            pt_C = [2600, 1370]
            pt_D = [1655, 365]
        case 2:
            pt_A = [1090, 310]
            pt_B = [0, 1310]
            pt_C = [2687, 1310]
            pt_D = [1680, 300]
        case 7:
            pt_A = [1185, 300]
            pt_B = [590, 1340]
            pt_C = [2684, 1340]
            pt_D = [1696, 325]
        case 3:
            pt_A = [1000, 335]
            pt_B = [0, 1370]
            pt_C = [2440, 1370]
            pt_D = [1600, 335]
        case 5:
            pt_A = [1055, 500]
            pt_B = [0, 1450]
            pt_C = [1970, 1450]
            pt_D = [1470, 515]

    if show:
        cv.circle(img, tuple(pt_A), 2, (0, 0, 255), 2)
        cv.circle(img, tuple(pt_B), 2, (0, 0, 255), 2)
        cv.circle(img, tuple(pt_C), 2, (0, 0, 255), 2)
        cv.circle(img, tuple(pt_D), 2, (0, 0, 255), 2)

        cv.line(img, pt_A, pt_B, (0, 0, 255), 2)
        cv.line(img, pt_B, pt_C, (0, 0, 255), 2)
        cv.line(img, pt_C, pt_D, (0, 0, 255), 2)
        cv.line(img, pt_D, pt_A, (0, 0, 255), 2)

        show_img(f"test{camera_nr}", img)

    # Used L2 norm to calculate distance between points
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])  # pyright: ignore
    output_pts = np.float32(
        [
            [0, 0],  # pyright: ignore
            [0, maxHeight - 1],
            [maxWidth - 1, maxHeight - 1],
            [maxWidth - 1, 0],
        ]
    )

    return (
        cv.getPerspectiveTransform(input_pts, output_pts),
        maxWidth,
        maxHeight,
    )


def get_bird_view(img, camera_nr, show=False):
    """
    Get a bird view of a given img matrix based on manually saved points

    Keyword arguments:
    img       -- undistorted image matrix to transform
    camera_nr -- camera number to get che correct points for perspective transform
    show      -- show trapezium

    Return a bird view image
    """

    M, maxWidth, maxHeight = get_bird_view_perspective_transform_matrix(
        camera_nr, show=show, img=img
    )

    return cv.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv.INTER_LINEAR)


def get_bird_view_points(camera_nr, und_pts):
    """
    Transform the undistorted points into bird view points
    """

    M, _, _ = get_bird_view_perspective_transform_matrix(camera_nr)

    return cv.perspectiveTransform(np.array([und_pts], dtype=np.float32), M).reshape(
        und_pts.shape[0], 2
    )


def transform_point_bird_to_undistort(camera_nr, point):
    M, _, _ = get_bird_view_perspective_transform_matrix(camera_nr)

    return cv.perspectiveTransform(np.array([point]), np.linalg.pinv(M))

''' def detect_cows(bird_view_list, model, show_result=False):
    """
    Detect cows in bird's eye view images.

    Args:
        bird_view_list: List of bird's eye view images in BGR format.
        model: Cow detection machine learning model.
        show_result: Display images with detected cows if True.

    Returns:
        List: Predicted bounding boxes (x, y, width, height) for detected cows.
    """
    bird_view_img_pil = []
    
    # Convert BGR images to RGB format
    for img in bird_view_list:
        bird_view_img_pil.append(Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB)))

    # Make predictions using the model
    (
        predicted_bboxes,
        predicted_class_confidences,
        predicted_class_labels,
    ) = model.predict(bird_view_img_pil)

    # Optionally display the results
    if show_result:
        for i, img in enumerate(bird_view_list):
            show_image(img, predicted_bboxes[i])

    return predicted_bboxes '''

def get_bboxs_centers(bboxes):
    centers = []

    for bbox in bboxes:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        centers.append([cx, cy])

    return centers


def groundProjectPoint(camera_nr, mtx, dist, points):
    obj_points = []

    if camera_nr == 1:
        obj_points = pts_data.ch1_obj
    elif camera_nr == 4:
        obj_points = pts_data.ch4_obj
    elif camera_nr == 8:
        obj_points = pts_data.ch8_obj
    elif camera_nr == 6:
        obj_points = pts_data.ch6_obj

    und_pts = undistort_points(camera_nr, mtx, dist).astype(np.float32)

    newcameramtx, _ = cv.getOptimalNewCameraMatrix(
        mtx,
        dist,
        (CAMERA_WIDTH, CAMERA_HEIGHT),
        UNDISTORTION_COEFFICENT,
        (CAMERA_WIDTH, CAMERA_HEIGHT),
    )

    _, rvecs, tvecs = cv.solvePnP(obj_points, und_pts, newcameramtx, 0)

    rotMat, _ = cv.Rodrigues(rvecs)

    camera_position = -np.matrix(rotMat).T * np.matrix(tvecs)

    real_pts = []

    for p in points:
        z = 100

        camMat = np.asarray(newcameramtx)
        iRot = inv(rotMat)
        iCam = inv(camMat)

        uvPoint = np.ones((3, 1))

        # Image point
        uvPoint[0, 0] = p[0]
        uvPoint[1, 0] = p[1]

        tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
        tempMat2 = np.matmul(iRot, tvecs)

        s = (z + tempMat2[2, 0]) / tempMat[2, 0]
        wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvecs))

        # wcPoint[2] may differ from z by tiny numerical error; allow small tolerance
        if abs(float(wcPoint[2] - z)) > 1e-4:
            # optional: log or warn if desired
            pass
        wcPoint[2] = z

        real_p = wcPoint.reshape(-1).astype(np.int32)

        # Check if predicted point overflow real barn dimensions and if fix it
        if real_p[0] > 4200:
            real_p[0] = 4200
        if real_p[0] < 0:
            real_p[0] = 0
        if real_p[1] > 2950:
            real_p[1] = 2950
        if real_p[1] < 0:
            real_p[1] = 0

        real_pts.append(real_p)

    return real_pts


def testGroundProjectPoint(img, mtx, dist, top_view_points, indexes, z=180.0):
    obj_points = pts_data.ch1_obj
    img_points = pts_data.ch1_img

    und_pts1 = undistort_points(1, mtx, dist).astype(np.float32)

    # interested_point_indexes = [0, 27, 37, 60]
    # h, status = cv.findHomography(
    #    und_pts1[indexes][interested_point_indexes].astype(np.int32),
    #    top_view_points[interested_point_indexes],
    #    cv.RANSAC,
    # )

    newcameramtx, _ = cv.getOptimalNewCameraMatrix(
        mtx,
        dist,
        (CAMERA_WIDTH, CAMERA_HEIGHT),
        UNDISTORTION_COEFFICENT,
        (CAMERA_WIDTH, CAMERA_HEIGHT),
    )

    # These are for the bird view
    # mse 9
    # newcameramtx[0][0] = 537
    # newcameramtx[1][1] = 342
    # newcameramtx[0][2] = 1300
    # newcameramtx[1][2] = 722

    # mse 9 tuned
    # newcameramtx[0][0] = 467
    # newcameramtx[1][1] = 294
    # newcameramtx[0][2] = 1324
    # newcameramtx[1][2] = 746

    result, rvecs, tvecs = cv.solvePnP(
        obj_points[indexes], und_pts1[indexes], newcameramtx, 0
    )

    rotMat, _ = cv.Rodrigues(rvecs)

    # out_pts, jacobian = cv.projectPoints(
    #    obj_points[indexes], rvecs, tvecs, newcameramtx, 0
    # )
    # out_pts = out_pts.reshape(out_pts.shape[0], 2)

    # mse = cv.norm(top_view_points, out_pts, cv.NORM_L2) / len(top_view_points)

    # for p in out_pts:
    #    cv.circle(img, p.astype(np.int32), 3, (0, 0, 255), 5)

    # show_img("Reprojection", img)
    # print(f"MSE: {mse}")

    # These are the correct cm
    camera_position = -np.matrix(rotMat).T * np.matrix(tvecs)
    print(
        f"Camera position:\nX:{camera_position[0]}\nY:{camera_position[1]}\nZ:{camera_position[2]}\n"
    )

    # interested_topdown_point_indexes = [0, 8, 27, 34, 47, 60]
    # z = 135.0
    # i = 47

    real_pts = []

    for i in range(len(und_pts1[indexes])):
        z = obj_points[indexes][i][2]
        image_point = und_pts1[indexes][i]

        camMat = np.asarray(newcameramtx)
        iRot = inv(rotMat)
        iCam = inv(camMat)

        uvPoint = np.ones((3, 1))

        # Image point
        uvPoint[0, 0] = image_point[0]
        uvPoint[1, 0] = image_point[1]

        tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
        tempMat2 = np.matmul(iRot, tvecs)

        s = (z + tempMat2[2, 0]) / tempMat[2, 0]
        wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvecs))

        # wcPoint[2] will not be exactly equal to z, but very close to it
        assert int(abs(wcPoint[2] - z) * (10**8)) == 0
        wcPoint[2] = z

        real_pts.append(wcPoint.reshape(-1).astype(np.int32))

        print(f"Point {i} with z={z}")
        print(
            f"Calculated: {wcPoint.reshape(-1).astype(np.int32)} \nExpected: {obj_points[indexes][i]}"
        )

    mse = cv.norm(
        obj_points[indexes].astype(np.int32), np.array(real_pts), cv.NORM_L2
    ) / len(und_pts1[indexes])
    print(mse)


def merge_duplicate_points(points, treshold=40):
    points_interested_index = []

    for i, p in enumerate(points):
        # Take only points inside the roi
        if (p[0] > 2200 and p[0] < 2750) or (p[1] > 1200 and p[1] < 1700):
            points_interested_index.append(i)

    to_remove = []
    for i, p1 in enumerate(points):
        for p2 in points[points_interested_index]:
            if not (p1 == p2).all():
                dist = math.dist(p1, p2)
                if dist <= treshold:
                    to_remove.append(i)

    mask = np.ones(len(points), dtype=bool)
    mask[to_remove] = False
    return points[mask]


def points_to_barn(points, barn_img, show=False):
    """
    Plot a real world point into the barn image

    Keyword arguments:
    point -- real world 3d point to project
    """
    barn_real_height, barn_real_width = 2950, 4200  # in centimeters
    img_height, img_widht = barn_img.shape[:2]

    # Calculate the pixel/cm ratio
    pixel_to_cm_height, pixel_to_cm_width = (
        barn_real_height / img_height,
        barn_real_width / img_widht,
    )

    barn_points = []
    for p in points:
        # OpenCV start from top-left and our 3d points start from bottom-left
        # Solved this substracting the caclulated height with the image one
        img_px_x, img_px_y = (
            p[0] / pixel_to_cm_width,
            img_height - (p[1] / pixel_to_cm_height),
        )
        barn_points.append((int(img_px_x), int(img_px_y)))

        cv.circle(barn_img, (int(img_px_x), int(img_px_y)), 2, (162, 81, 250), 4)

    if show:
        show_img("barn", barn_img)

    return barn_points, barn_img


def get_bird_view_full(img, camera_nr):
    """
    WIP function to obtain bird view without crop
    """
    # specify input coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x,
    if camera_nr == 1:
        pt_A = [960, 290]
        pt_B = [0, 1370]
        pt_C = [2600, 1370]
        pt_D = [1670, 290]
    elif camera_nr == 4:
        pt_A = [965, 370]
        pt_B = [0, 1370]
        pt_C = [2687, 1370]
        pt_D = [1694, 370]
    elif camera_nr == 6:
        pt_A = [1000, 335]
        pt_B = [0, 1370]
        pt_C = [2687, 1370]
        pt_D = [1694, 335]
    elif camera_nr == 8:
        pt_A = [1045, 365]
        pt_B = [0, 1370]
        pt_C = [2600, 1370]
        pt_D = [1655, 365]

    hh, ww = img.shape[:2]

    input = np.array([pt_A, pt_D, pt_C, pt_B], dtype=np.float32)

    # get top and left dimensions and set to output dimensions of red rectangle
    width = round(math.hypot(input[0, 0] - input[1, 0], input[0, 1] - input[1, 1]))
    height = round(math.hypot(input[0, 0] - input[3, 0], input[0, 1] - input[3, 1]))

    # set upper left coordinates for output rectangle
    x = input[0, 0]
    y = input[0, 1]

    # specify output coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x,
    output = np.float32(
        [
            [x, y],
            [x + width - 1, y],
            [x + width - 1, y + height - 1],
            [x, y + height - 1],
        ]
    )

    # compute perspective matrix

    matrix = cv.getPerspectiveTransform(input, output)

    imgOutput = cv.warpPerspective(
        img,
        matrix,
        (ww, hh),
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_TRANSPARENT,
        borderValue=(0, 0, 0),
    )

    return imgOutput


"""
def test():
    img4 = cv.imread('assets/original_images/ch3.png')
    img1 = cv.imread('assets/original_images/ch1.png')
    img6 = cv.imread('assets/original_images/ch6.png')

    mtx, dist = ut.load_model(os.path.join('assets/camera_params/params.npz'))
    crop_data4 = np.load(os.path.join('assets/camera_params/crop/Ch4_crop_size.npz'))
    crop_data1 = np.load(os.path.join('assets/camera_params/crop/Ch1_crop_size.npz'))
    crop_data6 = np.load(os.path.join('assets/camera_params/crop/Ch6_crop_size.npz'))

    img4_undistorted = ut.undistort_rational(img4, (mtx, dist))
    img1_undistorted = ut.undistort_rational(img1, (mtx, dist))
    img6_undistorted = ut.undistort_rational(img6, (mtx, dist))

    img4_undistorted_cropped = img4_undistorted[crop_data4['crop'][2]:crop_data4['crop'][3],crop_data4['crop'][0]:crop_data4['crop'][1],:]
    img1_undistorted_cropped = img1_undistorted[crop_data1['crop'][2]:crop_data1['crop'][3],crop_data1['crop'][0]:crop_data1['crop'][1],:]
    img6_undistorted_cropped = img6_undistorted[crop_data6['crop'][2]:crop_data6['crop'][3],crop_data6['crop'][0]:crop_data6['crop'][1],:]

    #cv.namedWindow("1", cv.WINDOW_NORMAL)
    #cv.resizeWindow("1", 300, 700)
    #cv.imshow('4', img4_undistorted_cropped)
    #cv.imshow('1', img1_undistorted)
    #cv.imshow('6', cv.rotate(img6_undistorted_cropped, cv.ROTATE_180))

    pt_A = [1050, 575]
    pt_B = [300, 1050]
    pt_C = [2260, 1050]
    pt_D = [1510, 575]
"""