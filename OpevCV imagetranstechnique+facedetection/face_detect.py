
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#load the image to be analyzed
img = cv.imread('C:/Users/Dell/Desktop/Mlda/CV projects/Pureopencv/imagetranstechnique+facedetection/facedetect/groupphoto.jpeg')
cv.imshow('original image', img)



# Blurring examples
# Averaging
average = cv.blur(img, (3,3))
cv.imshow('Average Blur', average)
cv.imwrite('avgblur.jpg', average)


# Gaussian Blur
gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian Blur', gauss)
cv.imwrite('gaussblur.jpg', gauss)

# Median Blur
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)
cv.imwrite('medblur.jpg', median)


# Bilateral
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral', bilateral)
cv.imwrite('bilblur.jpg', bilateral)





# Color spaces
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite('gray.jpg', gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)
cv.imwrite('hsv.jpg', hsv)

# BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)
cv.imwrite('lab.jpg', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)
cv.imwrite('rgb.jpg', rgb)

# HSV to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB --> BGR', lab_bgr)





# Edge detection techniques
# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)
cv.imwrite('edge_lap.jpg', lap)

# Sobel 
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)
cv.imwrite('sobelx.jpg', sobelx)
cv.imwrite('sobely.jpg', sobely)
cv.imwrite('combined_sobel.jpg', combined_sobel)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)
cv.imwrite('canny.jpg', canny)







# histogram of RGB in the image
plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

# Save the plot
plt.savefig('rgbhist.png', dpi=300)  # Save as high-res image
plt.show()






# rescale and resize image
def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resized = rescaleFrame(img, scale=0.75)
cv.imshow('resized', resized)
cv.imwrite('resized.jpg', resized)


def changeRes(width,height):
    # Live video
    capture.set(3,width)
    capture.set(4,height)
    
# Reading Videos
# capture = cv.VideoCapture('<your path here>/filename.mp4')

# while True:
#     isTrue, frame = capture.read()

#     frame_resized = rescaleFrame(frame, scale=.2)
    
#     cv.imshow('Video', frame)
#     cv.imshow('Video Resized', frame_resized)

#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break

# capture.release()
# cv.destroyAllWindows()





### splitting and merging color channels
blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r = cv.split(img)

blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])
cv.imwrite('blue.jpg', blue)
cv.imwrite('green.jpg', green)
cv.imwrite('red.jpg', red)



cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b,g,r])
cv.imshow('Merged Image', merged)





#Face detection with built in modules in opencv
haar_cascade = cv.CascadeClassifier('haar_face.xml')

print(haar_cascade)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)

print(faces_rect)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)
cv.imwrite('detected_faces.jpg', img)



cv.waitKey(0)