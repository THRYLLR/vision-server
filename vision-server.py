#!/usr/bin/env python3

import time
import argparse
import ntcore

import cv2 as cv
import numpy as np
import pupil_apriltags

def setPoseTables(rot, trans, rvec, tvec):
  rot.putNumber("X", rvec[0][0])
  rot.putNumber("Y", rvec[1][0])
  rot.putNumber("Z", rvec[2][0])

  trans.putNumber("X", tvec[0][0])
  trans.putNumber("Y", tvec[1][0])
  trans.putNumber("Z", tvec[2][0])

point = lambda t: (int(t[0][0]), int(t[0][1]))
green = (0, 255, 0)

cameraMat = np.matrix(
  [[604.63310129,   0,         261.83745148],
  [  0,         600.52960814, 248.59373312],
  [  0,           0,           1        ]]
)

distCoeffs = np.matrix([[
  0.07758159, 0.0582485, 0.0021086, -0.0304751, -0.42220893
]])

objectPoints = np.matrix([
  [-3, 3, 0],
  [3, 3, 0],
  [3, -3, 0],
  [-3, -3, 0]
], dtype = "double")

parser = argparse.ArgumentParser(
  description = "AprilTag pose estimation solution meant to run on an NVIDIA Jetson Nano.",
  add_help = True
)

parser.add_argument("-s", "--server")
parser.add_argument("-c", "--camera_id")
arguments = parser.parse_args()

inst = ntcore.NetworkTableInstance.getDefault()
inst.startClient3("VisionServer")
inst.setServer(arguments.server)

network = inst.getTable("SmartDashboard").getSubTable("VisionServer")
rotTable = network.getSubTable("Rot")
transTable = network.getSubTable("Trans")

cap = cv.VideoCapture(int(arguments.camera_id))

detector = pupil_apriltags.Detector(
  families = "tag36h11",
  nthreads = 6,
  quad_decimate = 2.0, # reduces resolution of detection input, improves speed but reduces accuracy
  quad_sigma = 0.0, # blurs detection input, reduces noise
  refine_edges = 1,
  decode_sharpening = 0.25 # sharpens detection input, good for small tags but bad for low light
)

while True:
  ret, image = cap.read()
  results = detector.detect(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

  if len(results) > 0:
    ret, rotationVec, translationVec = cv.solvePnP(objectPoints, results[0].corners, cameraMat, distCoeffs)
    setPoseTables(rotTable, transTable, rotationVec, translationVec)
  else:
    setPoseTables(rotTable, transTable, [[0], [0], [0]], [[0], [0], [0]])

  if cv.waitKey(30) == 1: break