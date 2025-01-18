import streamlit as st
from PIL import Image
import cv2
import numpy as np


Body_Parts = {
    'nose': 0, 'neck': 1, 'rshoulder': 2, 'relbow': 3, 'rwrist': 4,
    'lshoulder': 5, 'lelbow': 6, 'lwrist': 7, 'rhip': 8, 'rknee': 9,
    'rankle': 10, 'lhip': 11, 'lknee': 12, 'lankle': 13, 'reye': 14,
    'leye': 15, 'rear': 16, 'lear': 17, 'Background': 18
}

Pose_parts = [
    ['neck', 'rshoulder'], ['neck', 'lshoulder'], ['rshoulder', 'relbow'],
    ['relbow', 'rwrist'], ['lshoulder', 'lelbow'], ['lelbow', 'lwrist'],
    ['neck', 'rhip'], ['rhip', 'rknee'], ['rknee', 'rankle'], ['neck', 'lhip'],
    ['lhip', 'lknee'], ['lknee', 'lankle'], ['neck', 'nose'], ['nose', 'reye'],
    ['reye', 'rear'], ['nose', 'leye'], ['leye', 'lear']
]


inwidth = 368
inheight = 368
net = cv2.dnn.readNetFromTensorflow("D:\\BearandBee\\r\\pose\\graph_opt.pb")
st.title('Human Pose Estimation')
st.text('post a good image, not a damaged , unclear one')

img_file = st.file_uploader('Upload_Image',type=['jpg','png','jpeg','pdf'])
if img_file is not None:
    image = np.array(Image.open(img_file))
else:
    image = np.array(Image.open(r'D:\BearandBee\r\pose\runner.jpeg'))

    
st.subheader('Original Image')
st.image(image,caption='original Image',use_column_width=True)

thresh = st.slider('Threshold:',min_value=0,value=20,max_value=90,step=5)
thresh = thresh/100

@st.cache_resource
def posedetector(frame):
    framewidth = frame.shape[1]
    frameheight = frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inwidth, inheight), 
                                       (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    assert(len(Body_Parts) == out.shape[1])

    points = []
    for i in range(len(Body_Parts)):
        heatmap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        x = (framewidth * point[0]) / out.shape[3]
        y = (frameheight * point[0]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thresh else None)

    for pair in Pose_parts:
        partfrom = pair[0]
        partto = pair[1]
        assert(partfrom in Body_Parts)
        assert(partto in Body_Parts)

        idfrom = Body_Parts[partfrom]
        idto = Body_Parts[partto]
        if points[idfrom] and points[idto]:
            cv2.line(frame, points[idfrom], points[idto], (0, 225, 0), 3)
            cv2.ellipse(frame, points[idfrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idto], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    return frame
output = posedetector(image)
st.subheader('Position Estimated')
st.image(output,caption='position_estimated',use_column_width=True)

st.markdown('''
            #
            ''')

