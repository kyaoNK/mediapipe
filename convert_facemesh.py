from mediapipe.framework.formats.landmark_pb2 import LandmarkList
from mediapipe.framework.formats.rect_pb2 import NormalizedRect
from google.protobuf.json_format import MessageToJson

import glob
import re
from pprint import pprint
import json
import sys

if len(sys.argv) > 1:
    targetDir = "/mnt/c/Users/nkoji/main/research/video/result/" + sys.argv[1]
else:
    targetDir = "/mnt/c/Users/nkoji/main/research/video/result/WIN_20201026_14_46_53_Pro_Output/face_mesh/landmarks"

outputFiles = glob.glob(targetDir + "/" + "*.txt")

landmarkFiles = [(re.findall(
    r"iLoop=(\d+)_landmark_j=(\d+).txt", outputFile), outputFile) for outputFile in outputFiles]
landmarkFilesFiltered = [
    (landmarkFile[0], landmarkFile[1].replace("\\", "/")) for landmarkFile in landmarkFiles if landmarkFile[0]]


for landmarkFile in landmarkFilesFiltered:
    with open(landmarkFile[1], "rb") as f:
        content = f.read()

    landmark = LandmarkList()
    landmark.ParseFromString(content)
    jsonObj = MessageToJson(landmark)

    landmarkFileOutput = landmarkFile[1].replace("txt", "json")
    with open(landmarkFileOutput, "w") as f:
        f.write(jsonObj)