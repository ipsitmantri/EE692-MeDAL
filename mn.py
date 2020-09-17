import torch
import torchvision as vision
import PIL.Image
rc = vision.transforms.RandomCrop([224,224])
rr = vision.transforms.RandomRotation(degrees=30)
rhf = vision.transforms.RandomHorizontalFlip()
rvf = vision.transforms.RandomVerticalFlip()
cj = vision.transforms.ColorJitter()
input = PIL.Image.open('C:\Spring 2020\EE 692 - Amit Sethi\dataset_TMA\TMA_images\ZT76_39_A_1_1.jpg')
out1 = rc(input)
out2 = rr(out1)
out2 = cj(rvf(rhf(out2)))
out2.show()
out1.show()
