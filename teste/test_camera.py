from djitellopy import tello
import cv2

me = tello.Tello()

me.connect()

print(me.get_battery())

me.streamon()
while True:
    img = me.get_frame_read().frame
    print('Original Dimensions : ', img.shape)

    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)

    cv2.imshow("hello", img)
    cv2.imshow("Resized image", resized)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        opreste=True

me.streamoff()
cv2.destroyAllWindows()
