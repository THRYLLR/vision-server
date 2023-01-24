#!/usr/bin/env python3

from argparse import ArgumentParser

parser = ArgumentParser(
  description = "AprilTag pose estimation solution meant to run on an NVIDIA Jetson Nano.",
  add_help = True
)

parser.add_argument("-t", "--team_number")
parser.add_argument("-c", "--camera_id")