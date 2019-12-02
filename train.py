from time import time
import  vgg16 as model
import utils as render
import tensorflow as tf
import  os
import warnings
warnings.filterwarnings("ignore")

startTime=time()
batch_size=32
capacity=256
means=[123.68,116.779,103.939]
xs,ys=render.get_file("D:/deeplearning/dogcat/train/")
image_batch,label_batch=render.get_batch(xs,ys,224,224,batch_size,capacity)

x=tf.placeholder(tf.float32,[None,224,224,3])
y=tf.placeholder(tf.int32,[None,2])

vgg=model.vgg16(x)
fc8_finetuining=vgg.probs
#loss
loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc8_finetuining,labels=y))
#优化器
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_function)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
vgg.load_weights('D:/deeplearning/dogcat/vgg16_weights.npz',sess)
saver=tf.train.Saver()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
epoch_start_time = time()
for i in range(20):
    images, labels = sess.run([image_batch, label_batch])
    labels = render.one_hot(labels)

    sess.run(optimizer, feed_dict={x: images, y: labels})
    loss = sess.run(loss_function, feed_dict={x: images, y: labels})
    print("Now the loss is %f" % loss)

    epoch_end_time = time()
    print("Current epoch takes:",(epoch_end_time - epoch_start_time))
    epoch_start_time = epoch_end_time

    if (i + 1) % 500 == 0:
        saver.save(sess, os.path.join("D:/deeplearning/dogcat/code/model/", 'epoch{:06d}.ckpt'.format(i)))
    print("--------------------------------epoch %d is finished")


saver.save(sess,"D:/deeplearning/dogcat/code/model/")
print("optimizer finished")

duration=time()-startTime
print ("Train finished takes:","{:.2f}".format(duration))

coord.request_stop()
coord.join(threads)
