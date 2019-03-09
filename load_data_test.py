import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


test_images = pd.read_pickle(
    '/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/test_images.pkl')
new_img = test_images.reshape(
    test_images.shape[0], (test_images.shape[1]*test_images.shape[2]))
np.savetxt(fname='/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/test/images.csv',
           X=new_img, delimiter=',', fmt='%d')
