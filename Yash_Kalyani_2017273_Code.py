import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import imageio

#----------Question 1----------
def getHistogram(inp_img, name):
	m, n = inp_img.shape

	x_hist = range(0, 256)
	y_hist = [0 for i in x_hist]

	for y in range(0, m):
		for x in range(0, n):
			y_hist[int(inp_img[y][x])] += 1

	fig, axes = plt.subplots(1, 2)
	plt.gray()
	axes[0].imshow(inp_img)
	axes[0].set_title(name + " Image")
	axes[1].bar(x_hist, y_hist)
	axes[1].set_title("Corresponding Histogram")
	plt.show()

	return np.array(y_hist)/(m*n)

def getCDF(pdf):
	CDF = []
	pSum = 0
	for p in pdf:
		pSum += p
		CDF.append(pSum)
	return np.array(CDF)

def q1():
	L = 256

	inp_img = imageio.imread("Yash_Kalyani_2017273_cameraman.tif")
	m, n = inp_img.shape
	P_r = getHistogram(inp_img, "Input")

	spec_img = imageio.imread("Yash_Kalyani_2017273_specified image.jpg")
	P_z = getHistogram(spec_img, "Specified")

	T_r = (L-1) * getCDF(P_r)
	G_z = (L-1) * getCDF(P_z)

	out_img = np.zeros((m, n))

	matching = {}
	idx = 0
	for r in T_r:
		matching[idx] = (np.abs(G_z - r)).argmin()
		idx += 1

	# print("Mapping is: ")
	# print(matching)

	for y in range(m):
		for x in range(n):
			out_img[y][x] = matching[inp_img[y][x]]
	getHistogram(out_img, "Output")

#----------Question 3a---------
def getTransposedFilter(kernel):
	m, n = kernel.shape
	img_kernel = np.zeros((m, n))

	for i in range(0, m):
		for j in range(0, n):
			img_kernel[m-i-1][n-j-1] = kernel[i][j]
	
	return img_kernel

def convolve(inp_img, kernel):
	m1, n1 = inp_img.shape
	m2, n2 = kernel.shape

	padded_inp = np.zeros((m1+m2-1, n1+n2-1))
	padded_inp[m2//2:m1+(m2//2), n2//2:n1+(n2//2)] = inp_img

	final_out = np.zeros((m1, n1))

	for y in range(m1):
		for x in range(n1):
			kersum = 0
			for i in range(m2):
				for j in range(n2):
					kersum += (padded_inp[y+i][x+j] * kernel[i][j])
			final_out[y][x] = kersum

	return final_out

def q3a():
	kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
	inp = np.array([[6,7],[8,9]])
	print("Convolution result for Q3a is: ")
	print(convolve(inp, getTransposedFilter(kernel)))

#----------Question 4----------
def getWKernel():
	# laplacian_filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])
	# m, n = laplacian_filter.shape
	
	# delta_xy = np.zeros(laplacian_filter.shape)
	# delta_xy[m//2][n//2] = 1
	
	# w_xy = np.zeros(laplacian_filter.shape)
	# if laplacian_filter[m//2][n//2] < 0:
	# 	w_xy = delta_xy - laplacian_filter
	# else:
	# 	w_xy = delta_xy + laplacian_filter

	w_xy_calc = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
	
	return w_xy_calc

def sharpen(inp_img):
	out_img = convolve(inp_img, getWKernel())

	fig, axes = plt.subplots(1, 2)
	plt.gray()
	axes[0].imshow(inp_img, vmin=0, vmax=255)
	axes[0].set_title("Original Image")
	axes[1].imshow(out_img, vmin=0, vmax=255)
	axes[1].set_title("Sharpened Image")
	plt.show()

def q4():
	inp_img = imageio.imread("Yash_Kalyani_2017273_chandrayaan.jpg")
	print("w(x, y) kernel is: ")
	print(getWKernel())
	sharpen(inp_img)

if __name__ == '__main__':

	q1()
	q3a()
	q4()