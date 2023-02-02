#!/usr/bin/env python3

# import ntcore
from networktables import NetworkTables
from argparse import ArgumentParser
import cv2 as cv

from estimator import Estimator
import cameras

def updateNetworkTag(tagTable, i, r, t):
  table = tagTable.getSubTable(str(i))
  rot = table.getSubTable("Rot")
  trans = table.getSubTable("Trans")

  rot.putNumber("X", r[0][0])
  rot.putNumber("Y", r[1][0])
  rot.putNumber("Z", r[2][0])

  trans.putNumber("X", t[0][0])
  trans.putNumber("Y", t[1][0])
  trans.putNumber("Z", t[2][0])

parser = ArgumentParser(
  description = "AprilTag pose estimation solution meant to run on an NVIDIA Jetson Nano.",
  add_help = True
)

parser.add_argument("-s", "--server")
parser.add_argument("-c", "--camera_id")
parser.add_argument("--nt3", action = "store_true")
args = parser.parse_args()

NetworkTables.initialize(server = args.server)
mainTable = NetworkTables.getTable("SmartDashboard/VisionServer")
tagTable = mainTable.getSubTable("Tags")

video = cv.VideoCapture(int(args.camera_id))
poseEstimator = Estimator(cameras.WEIRD_USB_CAMERA)

for tag in tagTable.getSubTables():
  updateNetworkTag(tagTable, tag, [[0], [0], [0]], [[0], [0], [0]])

while True:
  ret, image = video.read()
  image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  tagPoses = poseEstimator.estimate(image)

  for i, (r, t) in enumerate(tagPoses):
    updateNetworkTag(tagTable, i, r, t)

  if cv.waitKey(30) == 1: break