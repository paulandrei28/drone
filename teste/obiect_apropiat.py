import cv2
import torch
import time
import numpy as np

def empty(a):
    pass


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



cv2.namedWindow("HSV")

cv2.resizeWindow("HSV", 640, 240)

cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)

cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)

cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)

cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)

cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)

cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)


# Open up the video capture from a webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():

    success, img = cap.read()
    y = 60
    x = 80
    h = 350
    w = 450
    #crop img
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

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)

    imgHsv = cv2.cvtColor(depth_map, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")

    h_max = cv2.getTrackbarPos("HUE Max", "HSV")

    s_min = cv2.getTrackbarPos("SAT Min", "HSV")

    s_max = cv2.getTrackbarPos("SAT Max", "HSV")

    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")

    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    lower = np.array([h_min, s_min, v_min])

    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(imgHsv, lower, upper)

    result = cv2.bitwise_and(img, img, mask=mask)

    print(f'[{h_min},{s_min},{v_min},{h_max},{s_max},{v_max}]')

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cv2.imshow("ceva", img)
    cv2.imshow("alceva", mask)
    cv2.imshow("altceva", result)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()