import cv2 as cv
import numpy as np

class Estimator:
  def __init__(self, camera):
    self.matrix = np.matrix(camera.matrix)

    self.distCoeffs = np.matrix(camera.distCoeffs)

    self.objectPoints = np.matrix([
      [-3, 3, 0],
      [3, 3, 0],
      [3, -3, 0],
      [-3, -3, 0]
    ], dtype = "double")

    self.detector = cv.aruco.ArucoDetector(
      dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11),
      detectorParams = cv.aruco.DetectorParameters()
    )

  def estimate(self, image):
    out = []

    corners, ids, rejected = self.detector.detectMarkers(image)
    for corner in corners:
      ret, rvec, tvec = cv.solvePnP(self.objectPoints, corner, self.matrix, self.distCoeffs)
      out.append((rvec, tvec))

    return out