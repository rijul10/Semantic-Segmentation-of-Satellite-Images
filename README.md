# Semantic-Segmentation-of-Satellite-Images
This was a contest taken from Codalabs which required the user to semanticalys segment the tiff formatted satellite images into 6 classes. The code for which is given below.

## Baby steps

The first step was data pre-processing or in other words - how to handle 2783 RGB images of 1024x1024 resolution each.
Data preprocessing for feature extraction

Instead of just rescaling the 1024x1024 images, two seperate lists were created. The first contained the rescaled images of 1024x1024 into 128x128 resolution for faster processing. The other list contained the images sliced using a stride of 16 and then rescaled to 128x128 resolution. This resulted in a better mean IOU score.

def crops(a, crop_size = 128):

    stride = 16

    croped_images = []
    h, w, c = a.shape

    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    # Slicing the image into 128*128 crops with a stride of 16
    for i in range(1):
        for j in range(1):
            crop_x = a[(i*stride):((i*stride)+crop_size), (j*stride):((j*stride)+crop_size), :]
            croped_images.append(crop_x)
    return croped_images

## Training

Model used - UNET.
Training the model was done on Nvidia Tesla DGX V100 which delivers 4X faster training than other GPU-based systems by using the NVIDIA GPU Cloud Deep Learning Stack with optimized versions of today’s most popular frameworks. NVIDIA. 10 epochs were used and a batch size of 32 used.
Results

## Results

![Results](https://github.com/rijul10/Semantic-Segmentation-of-Satellite-Images/blob/master/Capture.PNG)

Mean-iou obtained for the UNET architecture = 0.50

## Future Work

-Better feature extraction methods can be used - i.e creating individual masks for each classes and then training the model.
-Due to the class imbalance, i.e too much tree cover but very less water cover the average mean IOU was suffering. Balancing the classes before feeding into the network will help tons.

![Class](https://github.com/rijul10/Semantic-Segmentation-of-Satellite-Images/blob/master/Capture1.PNG)
