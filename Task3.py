
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def generateSobelFilterGx():
    sobelFilterGx = [[-1,0,1],
                     [-2,0,2],
                     [-1,0,1]]
    return sobelFilterGx


# In[3]:


def generateSobelFilterGy():
    sobelFilterGy = [[-1,-2,-1],
                     [0,0,0],
                     [1,2,1]]
    return sobelFilterGy


# In[4]:


def generate_laplace_filter():
    laplace_filter = [[0,-1,0],
                     [-1,8,-1],
                     [0,-1,0]]
    return np.asarray(laplace_filter)


# In[5]:


def generateGradientX(x, y, padded_input_image):
    sobelFilterGx = generateSobelFilterGx()
    gradientX = sum([sum([padded_input_image[i][j] * sobelFilterGx[i][j] for j in range(len(sobelFilterGx))]) for i in range(len(sobelFilterGx))])
    return int(gradientX)


# In[6]:


def generateGradientY(x, y, padded_input_image):
    sobelFilterGy = generateSobelFilterGy()
    gradientY = sum([sum([padded_input_image[i][j] * sobelFilterGy[i][j] for j in range(len(sobelFilterGy))]) for i in range(len(sobelFilterGy))])
    return gradientY


# In[7]:


def generate_laplace_output(x, y, padded_input_image):
    laplace_filter = generate_laplace_filter()
    output = sum([sum([padded_input_image[i][j] * laplace_filter[i][j] for j in range(len(laplace_filter))]) for i in range(len(laplace_filter))])
    return output


# In[8]:


def computeMagnitude(edgeX, edgeY):
    return (((edgeX**2) + (edgeY**2))**0.5)


# In[9]:


def convolve(patch):
    convolved_patch = [patch[j][i] for j in range(len(patch[0])-1,-1,-1) for i in range(len(patch)-1,-1,-1)]
    return np.asarray([convolved_patch[i:i+len(patch)] for i in range(0, len(convolved_patch), len(patch))])


# In[10]:


color_image = cv.imread("hough.jpg")
hsv_input_image = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)
red_mask_upper = cv.inRange(hsv_input_image, (170, 50, 50), (180, 255, 255))
red_input_image = cv.bitwise_and(hsv_input_image,hsv_input_image, mask=red_mask_upper)
cv.imwrite("red_input_image.jpg",red_input_image)


# In[11]:


red_image = cv.imread("red_input_image.jpg",0)
padded_red_image = np.pad(red_image, ((1,1),(1,1)), 'edge')
red_edge_image = np.zeros(red_image.shape)


# In[12]:


for x in range(len(red_edge_image)):
    for y in range(len(red_edge_image[0])):
        patch = padded_red_image[x:x+3,y:y+3]
        localEdgeX = generateGradientX(x, y, patch)
        localEdgeY = generateGradientY(x, y, patch)
        red_edge_image[x][y] = computeMagnitude(localEdgeX,localEdgeY)

red_edge = np.zeros(red_edge_image.shape)
for x in range(len(red_edge)):
    for y in range(len(red_edge[0])):
        if(red_edge_image[x][y] >=255):
            red_edge[x][y] = 255
        else:
            red_edge[x][y] = 0
            
cv.imwrite("red_edge.jpg",red_edge)


# In[13]:


angle = np.arange(0,181)
output_image_shape = red_edge.shape
diagonal = np.ceil(((output_image_shape[0]**2)+(output_image_shape[1]**2)**0.5)).astype(int)
red_accumulator = np.zeros((diagonal, len(angle)))

for x in range(len(red_edge)):
    for y in range(len(red_edge[0])):
        if(red_edge[x,y] == 255):
            for every_angle in range(len(angle)):
                r = round((x * np.cos(angle[every_angle])) + (y * np.sin(angle[every_angle])), 2).astype(int)
                red_accumulator[r, every_angle] += 1                       


# In[18]:


T = 0.7 * np.max(red_accumulator)

red_rho_theta = []

for x in range(len(red_accumulator)):
    for y in range(len(red_accumulator[0])):
        if(red_accumulator[x][y] > T):
            red_rho_theta.append([x, angle[y]])


# In[19]:


original = cv.imread("hough.jpg")
for every_rho_theta in red_rho_theta:
    rh = every_rho_theta[0]
    thet = every_rho_theta[1]
    b = np.cos(thet)
    a = np.sin(thet)
    x0 = a*rh
    y0 = b*rh
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv.line(original,(x1,y1),(x2,y2),(0,255,0),2)
cv.imwrite("red_lines.jpg",original)


# In[22]:


color_image = cv.imread("hough.jpg")
hsv_input_image = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)
blue_mask_upper = cv.inRange(hsv_input_image, (100, 50, 50), (120, 255, 255))
blue_input_image = cv.bitwise_and(hsv_input_image,hsv_input_image, mask=blue_mask_upper)
cv.imwrite("blue_input_image.jpg",blue_input_image)


# In[23]:


blue_image = cv.imread("blue_input_image.jpg",0)
padded_blue_image = np.pad(blue_image, ((1,1),(1,1)), 'edge')
blue_edge_image = np.zeros(blue_image.shape)


# In[24]:


for x in range(len(blue_edge_image)):
    for y in range(len(blue_edge_image[0])):
        patch = padded_blue_image[x:x+3,y:y+3]
        localEdgeX = generateGradientX(x, y, patch)
        localEdgeY = generateGradientY(x, y, patch)
        blue_edge_image[x][y] = computeMagnitude(localEdgeX,localEdgeY)

blue_edge = np.zeros(blue_edge_image.shape)
for x in range(len(blue_edge)):
    for y in range(len(blue_edge[0])):
        if(blue_edge_image[x][y] >=255):
            blue_edge[x][y] = 255
        else:
            blue_edge[x][y] = 0
            
cv.imwrite("Blue_edge.jpg",blue_edge)


# In[25]:


angle = np.arange(0,181)
output_image_shape = blue_edge.shape
diagonal = np.ceil(((output_image_shape[0]**2)+(output_image_shape[1]**2)**0.5)).astype(int)
blue_accumulator = np.zeros((diagonal, len(angle)))

for x in range(len(blue_edge)):
    for y in range(len(blue_edge[0])):
        if(blue_edge[x,y] == 255):
            for every_angle in range(len(angle)):
                r = round((x * np.cos(angle[every_angle])) + (y * np.sin(angle[every_angle])), 2).astype(int)
                blue_accumulator[r, every_angle] += 1                       


# In[26]:


T = 0.3 * np.max(blue_accumulator)
blue_rho_theta = []

for x in range(len(blue_accumulator)):
    for y in range(len(blue_accumulator[0])):
        if(blue_accumulator[x][y] > T):
            blue_rho_theta.append([x, angle[y]])


# In[27]:


original = cv.imread("hough.jpg")
for every_rho_theta in blue_rho_theta:
    rh = every_rho_theta[0]
    thet = every_rho_theta[1]
    b = np.cos(thet)
    a = np.sin(thet)
    x0 = a*rh
    y0 = b*rh
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv.line(original,(x1,y1),(x2,y2),(0,255,255),2)
cv.imwrite("blue_lines.jpg",original)


# In[28]:


original_image = cv.imread("hough.jpg")
for every_rho_theta in blue_rho_theta:
    rh = every_rho_theta[0]
    thet = every_rho_theta[1]
    b = np.cos(thet)
    a = np.sin(thet)
    x0 = a*rh
    y0 = b*rh
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(original_image,(x1,y1),(x2,y2),(0,255,255),2)
for every_rho_theta in red_rho_theta:
    rh = every_rho_theta[0]
    thet = every_rho_theta[1]
    b = np.cos(thet)
    a = np.sin(thet)
    x0 = a*rh
    y0 = b*rh
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(original_image,(x1,y1),(x2,y2),(0,255,0),2)
cv.imwrite("all_lines.jpg",original_image)

