import cv2, sys, pytesseract
import numpy as np

def _exit():
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        sys.exit(0)


def putContNum(cont_num, x, y):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x+1,y+8)
    fontScale              = 0.3
    fontColor              = (0,0,0)
    lineType               = 2
      
    cv2.putText(imgcolor,
            "{}".format(cont_num),
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)


img = cv2.imread(sys.argv[1],0)

den = cv2.fastNlMeansDenoising(img, None, 21, 5, 21)
img = den
thresh = cv2.threshold(img,127,255,0)[1]
thfake = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
print('thresh', thresh)
print('thfake', thfake)
print(cv2.THRESH_BINARY_INV)
cv2.imshow('opencv-result', den)
cv2.imshow('opencv-resultor', img)
cv2.imshow('opencv-resulth', thresh)
#_exit()

imgcolor = cv2.imread(sys.argv[1])
ret,thresh = cv2.threshold(img,127,255,0)
imcont,contours,hierarchy = cv2.findContours(thresh, 1, 2)
cv2.imwrite('imcontsav.png',imcont)
imcontsavcolor = cv2.imread('imcontsav.png')

print("contours len", len(contours))
i = 0
areas = {}
for cont in contours:
    i+=1
    areas[i] = cv2.contourArea(cont)
    print("cont {} area = ".format(i), cv2.contourArea(cont))
    print(cont)

print('areas', areas)
areas_sorted = sorted(areas.items(), key=lambda x: x[1], reverse=True)
print('areas_sorted', areas_sorted)

for i in range(0,len(areas)):
    cont_num = areas_sorted[i][0] - 1
    print(i, areas_sorted[i], areas_sorted[i][0])
    print("cont {} area = ".format(cont_num), cv2.contourArea(contours[cont_num]))
    if cont_num != 999:
        x,y,w,h = cv2.boundingRect(contours[cont_num])
#        putContNum(cont_num,x,y)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.rectangle(imgcolor,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.rectangle(imcontsavcolor,(x,y),(x+w,y+h),(0,255,0),1)

def to_rgb(img):
    return cv2.merge([img] * 3)
'''
res = np.concatenate((
    to_rgb(img),
    to_rgb(imcont)))

cv2.imshow('opencv-result', res)
'''
cv2.imshow("img", img)
cv2.imshow("imgcolor", imgcolor)
cv2.imshow("imcont", imcont)
cv2.imshow("imcontsavcolor", imcontsavcolor)

print("Possible solustion is ", pytesseract.image_to_string(imcont, config='nobatch digits'))

k = cv2.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
