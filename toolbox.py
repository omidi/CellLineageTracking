from imports import *
display_min = 0
display_max = 65535
cols=512
rows=512

def convertTo8bit(image, display_min, display_max):
    # Here I set copy=True in order to ensure the original image is not
    # modified. If you don't mind modifying the original image, you can
    # set copy=False or skip this step.
    image = np.float16(image)
    image = np.array(image, copy=True)
    image.clip(display_min, display_max, out=image)
    image -= display_min
    image //= (display_min - display_max + 1) / 256.
    return 255 - image.astype(np.uint8)

def show_image(img):
	fig = plt.figure("image", figsize=(12,12))
	ax = fig.add_subplot(1,1,1)
	ax.imshow(img, cmap='gray')
	plt.axis("off")
	plt.show()

