import tensorflow as tf
import cv2
import vgg16 as model
import  numpy as np
import scipy

means=[123.68,116.779,103.939]
x=tf.placeholder(tf.float32,[None,224,224,3])

sess=tf.Session()
vgg=model.vgg16(x)
fc8_finetuining=vgg.probs

saver=tf.train.Saver()
print("model restoring")
saver.restore(sess,'D:/deeplearning/dogcat/code/model/')
filepath='D:/deeplearning/dogcat/test1/22.jpg'
img=scipy.ndimage.imread(filepath,mode='RGB')
img=scipy.misc.imresize(img,(224,224))
img=img.astype(np.float32)
for c in range(3):
    img[:,:,c]-=means[c]
prob=sess.run(fc8_finetuining,feed_dict={x:[img]})
max_index=np.argmax(prob)

if max_index==0:
    print("this is a cat possibility %.6f"%prob[:,0])
else:
    print("this is a dog possibility %.6f" % prob[:, 0])

