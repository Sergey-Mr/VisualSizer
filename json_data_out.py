import json
import cv2 

def get_points(file_name):
	f = open(file_name)

	data = json.load(f)

	for i in data['people']:
		points = i['pose_keypoints_2d']
		coordinates = []
		for point in points:
			coordinates.append(point)
            
		return coordinates


	f.close()

