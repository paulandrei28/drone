import cv2
import functii



def main():

    coord=[0]
    center=[]
    while(len(coord)<4):
        dist=functii.go_forward()
        coord.append(dist)
        print(dist)
        print("go left")

    cpy=coord.sort()
    center.append(cpy[0]/2)
    center.append(cpy[1]/2)
    print(coord)
    print(center)



if __name__ == '__main__':
    main()