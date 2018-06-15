import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


help(tf.constant)
a= tf.constant([1.0,2.0],name= "a")

b = tf.constant([2.0,3.0],name = "b")

result = a+ b

print("Before evaluate: result=",result)

sess = tf.Session()
with sess.as_default():
    print("After evaluate: result.eval()=",result.eval())
print("After evaluate: result=",result)

print("sess.run(result)=",sess.run(result))
print("result.eval(session=sess)=",result.eval(session=sess))


sess_iter = tf.InteractiveSession()

print("result.eval()=",result.eval())

sess_iter.close()

try:
    result.eval()
except Exception as e:
    print(e)

config = tf.ConfigProto(allow_soft_placement=True,log_device_placement= True)

sess1 = tf.Session(config= config)
sess2 = tf.InteractiveSession(config = config)

weights = tf.Variable(tf.random_normal([2,3],stddev=2))
print("weights=",weights)




w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
x = tf.constant([[0.7,0.9]])
a = tf.matmul(x,w1)
y= tf.matmul(a,w2)

sess = tf.Session()
sess.run(w1.initializer)
sess.run(w2.initializer)
print(sess.run(y))
sess.close()


sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(y))
sess.close()

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

LOG_DIR = os.path.join(PROJECT_DIR,'log/tf')


writer = tf.summary.FileWriter(LOG_DIR,tf.get_default_graph())
writer.close()
