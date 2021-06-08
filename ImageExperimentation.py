import tensorflow as tf
import re
import numpy as np
import os as os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from datetime import datetime
import sys

mypath1=input("Enter path 1: ")
mypath2=input("Enter path 2: ")
image_path1="ozge control 10 pa after_028.32_017.jpg"
image_path2="frac_avg1 (1).jpg"
image1=img.imread(fname=mypath1+"\\"+image_path1)
image2=img.imread(fname=mypath2+"\\"+image_path2)

image1_scaled=image1/255
image2_scaled=image2/255

print("something")