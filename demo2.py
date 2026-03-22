import cv2
import torch
import time
import numpy as np

valori_hsv=[0,0,0,179,143,255]
lower = np.array([valori_hsv[0], valori_hsv[1], valori_hsv[2]])
upper = np.array([valori_hsv[3], valori_hsv[4],valori_hsv[5]])

model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# # Open up the video capture from a webcam
# cap = cv2.VideoCapture(0)
from djitellopy import tello

me = tello.Tello()

me.connect()

print(me.get_battery())

me.streamon()

time.sleep(2)

me.takeoff()

opreste = False
count=0
while count <5:

    img = me.get_frame_read().frame
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply input transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)


    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    imgHsv = cv2.cvtColor(depth_map, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    y = 0
    x = 178
    h = 432
    w = 220
    # crop img
    mask_center = mask[y:y + h, x:x + w]
    y = 0
    x = 398
    h = 432
    w = 178
    # crop img
    mask_right = mask[y:y + h, x:x + w]
    y = 0
    x = 0
    h = 432
    w = 178
    # crop img
    mask_left = mask[y:y + h, x:x + w]

    # cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    number_of_white_pix = np.sum(mask_center == 255)
    print(number_of_white_pix/mask_center.size)

    left_pixels=np.sum(mask_left == 255)/mask_left.size
    right_pixels=np.sum(mask_right == 255)/mask_right.size

    me.move("forward",20)
    time.sleep(1)
    if number_of_white_pix/mask_center.size >=0.1:
        cv2.putText(result, f'Stop', (160,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        print("Stop")
        me.rotate_clockwise(90)
        count+=1

    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)
    cv2.imshow('Mask', mask)
    cv2.imshow('Centru', mask_center)
    cv2.imshow('stanga', mask_left)
    cv2.imshow('dreapta', mask_right)
    cv2.imshow("Close objects", result)

    if left_pixels > 0.1 or right_pixels > 0.1:
        count+=1
        if left_pixels > right_pixels:
            print("dreapta")
            me.rotate_clockwise(90)
        else:
            print("stanga")
            me.rotate_counter_clockwise(90)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

me.land()
me.streamoff()
me.end()
cv2.destroyAllWindows()