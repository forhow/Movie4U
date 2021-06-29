import cv2
import dlib

import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


'''
dlib 모델 다운로드
- https://github.com/davisking/dlib-models

참고 https://webnautes.tistory.com/1410

'''
print(cv2.__version__)
print(dlib.__version__)

# dlib.add_lib_to_dll_path()

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')
sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

img = dlib.load_rgb_image('./imgs/02.jpg')
plt.figure(figsize=(16,18))
# plt.imshow(img)
# plt.show()

'사람 얼굴에 박스 표시'
img_result = img.copy()

dets = detector(img)

print(dets)
# 너비 좌측 시작점, 높이 최고점, 너비, 높이를 반환

if len(dets) ==0:
    print('cannot find face')
else:
    # 한 사진안에 여러 얼굴이 있을 경우 모든 얼굴에 박스표시 위한 반복문
    fig, ax = plt.subplots(1, figsize=(16,10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y), w,h, linewidth=2, edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()

'이미지에서 Landmark 찾기'
fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y),
                                radius=3,
                                facecolor='r',
                                edgecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)
plt.show()

# 얼굴부분만 잘라서 보기
faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)
plt.show()

# # 이미지에서 얼굴부분만 추출해서 반환하는 함수 선언
# def align_faces(img):
#     dets = detector(img, 1)
#     objs = dlib.full_object_dectections()
#     for detection in dets:
#         s = sp(img, detection)
#         objs.append(s)
#     faces = dlib.get_
