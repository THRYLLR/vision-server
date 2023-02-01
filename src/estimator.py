import cv2 as cv
import numpy as np
# from pupil_apriltags import Detector

class Estimator:
  def __init__(self):
    self.matrix = np.matrix([
      [604.63310129, 0, 261.83745148],
      [0, 600.52960814, 248.59373312],
      [0, 0, 1]
    ])

    self.distCoeffs = np.matrix([[
      0.07758159, 0.0582485, 0.0021086, -0.0304751, -0.42220893
    ]])

    self.objectPoints = np.matrix([
      [-3, 3, 0],
      [3, 3, 0],
      [3, -3, 0],
      [-3, -3, 0]
    ], dtype = "double")

    # self.detector = Detector(
    #   families = "tag36h11",
    #   nthreads = 6,
    #   quad_decimate = 2.0, # reduces resolution of detection input, improves speed but reduces accuracy
    #   quad_sigma = 0.0, # blurs detection input, reduces noise
    #   refine_edges = 1,
    #   decode_sharpening = 0.25 # sharpens detection input, good for small tags but bad for low light
    # )

    self.detector = cv.aruco.ArucoDetector(
      dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11),
      detectorParams = cv.aruco.DetectorParameters()
    )

  def estimate(self, image):
    out = []

    # tags = self.detector.detect(image)
    corners, ids, rejected = self.detector.detectMarkers(image)
    for corner in corners:
      ret, rvec, tvec = cv.solvePnP(self.objectPoints, corner, self.matrix, self.distCoeffs)
      out.append((rvec, tvec))

    return out