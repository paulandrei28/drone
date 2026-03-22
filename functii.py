import cv2
import torch
import time
import numpy as np

from djitellopy import tello

me = tello.Tello()

me.connect()

print(me.get_battery())

me.streamon()
me.rotate_counter_clockwise(90)


valori_hsv=[0,0,0,118,255,255]
lower = np.array([valori_hsv[0], valori_hsv[1], valori_hsv[2]])
upper = np.array([valori_hsv[3], valori_hsv[4],valori_hsv[5]])




# Load a MiDas model for depth estimation
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def go_forward():
    distanta=0
    opreste = False
    while opreste is False:

        img = me.get_frame_read().frame
        cv2.imshow("salut", img)
        y = 60
        x = 80
        h = 350
        w = 450
        # crop img
        img = img[y:y + h, x:x + w]

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
        y = 85
        x = 130
        h = 200
        w = 200
        # crop img
        mask_center = mask[y:y + h, x:x + w]

        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        number_of_white_pix = np.sum(mask_center == 255)

        # print(number_of_white_pix/mask_center.size)
        time.sleep(1)
        distanta+=10
        if number_of_white_pix/mask_center.size >=0.4:
            cv2.putText(result, f'Stop', (160,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            opreste=True
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

