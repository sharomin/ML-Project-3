# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
#print(os.listdir("/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist"))

#Files are stored in pickle format.
#Load them like how you load any pickle. The data is a numpy array
train_images = pd.read_pickle(
    '/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/train_images.pkl')
train_labels = pd.read_csv(
    '/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/train_labels.csv')
train_images.shape

#Let's show image with id 16
img_idx = 108
plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
plt.imshow(train_images[img_idx])
print(train_images)