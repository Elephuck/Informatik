import tkinter as tk
from tkinter.filedialog import askdirectory
import threading
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import inspect
import ctypes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# this is data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# window 
window = tk.Tk()
window.title('LSTM')
window.geometry('1300x600')
t_Rnn = threading.Thread()
t_update = threading.Thread()


class Load_Lstm(object):

    def RNN_load(
            self,
            X, 
            weights,
            biases,
            n_inputs, n_steps, n_hidden_units):

        # hidden layer for input to cell
        # X(128 batch, 28 X 28)
        X = tf.reshape(X, [-1, n_inputs])
        X_in = tf.matmul(X, weights['in']+biases['in'])
        X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
        # cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            n_hidden_units, 
            forget_bias=1.0, 
            state_is_tuple=True
            )
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell,
            output_keep_prob=0.5
            )
        # # lstm cell is divided into two parts c_state  m_state
        _init_state = lstm_cell.zero_state(1, dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(
            lstm_cell, X_in, 
            initial_state=_init_state, 
            time_major=False
            )
        # hidden layer for output as the final results
        results = tf.matmul(states[1], weights['out']+biases['out'])
        return results

    def Loading_Rnn_Lstm(
            self,
            training_iters,
            batch_size,
            n_inputs,
            n_steps,
            n_classes,
            loadpath,
            loadImagepath
         ):

        imm = np.array(Image.open(loadImagepath).convert('L'))
        imm = imm/255
        imm_3 = Image.fromarray(imm)
        imm_4 = imm_3.resize([28, 28])
        im_array = np.array(imm_4)
        fs = im_array.reshape((-1, 28, 28))
        n_hidden_units = 128
        x = 0.001
        # plt.axis([0, training_iters, 0, 1])
        plt.ion()
        # pic figure
        # Placeholder
        x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        # y = tf.placeholder(tf.float32, [None, n_classes])
        loadpath = loadpath.replace(".index", "")
        weights = {
    
            'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),

            'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }
        biases = {

            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
        }
        
        pred = self.RNN_load(
            x,
            weights,
            biases,
            n_inputs, n_steps, n_hidden_units)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess, loadpath)
                load_Result = sess.run(
                    tf.argmax(pred, 1),
                    feed_dict={x: fs}
                )
                sess.close()
        return load_Result


class Lstm(object):

    training_iters = 40000
    batch_size = 128
    # window2.withdraw()
    n_inputs = 28
    n_steps = 28
    n_classes = 10
    training_iters2 = 0
    ti = time.time()
    currtime = str(ti)

    # Only Number
    def OnlyNumber(self, content):
        if content.isdigit() or content == "":
            return True
        else:
            return False
    
    OnlyNumberR = window.register(OnlyNumber)

    def UpdatePic(self, path, currtime):
        while(1):
            if(os.path.exists(path+"\Classification"+currtime+".png")):
                time.sleep(1)
                img = Image.open(path+"\Classification"+currtime+".png")
                photo = ImageTk.PhotoImage(img)
                Show_Image['image'] = photo
                time.sleep(3)
            else:
                time.sleep(3)     

    def RNN(
            self,
            X,
            weights,
            biases,
            n_inputs,
            n_steps, n_hidden_units, batch_size
         ):

        # hidden layer for input to cell
        # X(128 batch, 28 X 28)
        X = tf.reshape(X, [-1, n_inputs])
        X_in = tf.matmul(X, weights['in']+biases['in'])
        X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
        # cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            n_hidden_units, 
            forget_bias=1.0, 
            state_is_tuple=True
        )
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell,
            output_keep_prob=0.5
        )
        """
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell] * n_hidden_units,
            state_is_tuple=True
        )
        """
        # # lstm cell is divided into two parts c_state  m_state
        _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(
            lstm_cell, X_in, 
            initial_state=_init_state, 
            time_major=False
            )
        # hidden layer for output as the final results
        results = tf.matmul(states[1], weights['out']+biases['out'])
        return results

    def Lstm_RNN(
            self,
            training_iters,
            batch_size,
            n_inputs,
            n_steps,
            n_classes,
            path,
            currtime
         ):

        t = threading.currentThread()
        print('traing', training_iters)
        print('batch', batch_size)
        print('inputs', n_inputs)
        print('steps', n_steps)
        print('classes', n_classes)
        doc = open(path+'\Classification'+currtime+'.txt', 'w+')
        print(training_iters, file=doc)
        print(batch_size, file=doc)
        print(n_inputs, file=doc)
        print(n_steps, file=doc)
        print(n_classes, file=doc)
        lr = 0.001
        n_hidden_units = 128
        x = 0.001
        # plt.axis([0, training_iters, 0, 1])
        xs = [0, 0]
        ys = [0, 0]
        plt.ion()
        # pic figure
        # Placeholder
        x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        y = tf.placeholder(tf.float32, [None, n_classes])
        weights = {
    
            'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),

            'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }
        biases = {

            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
        }
        
        pred = self.RNN(
            x,
            weights,
            biases,
            n_inputs,
            n_steps,
            n_hidden_units,
            batch_size
            )
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=pred,
            labels=y
            ))
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            step = 0
            while step*batch_size < training_iters:
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
                sess.run([train_op], feed_dict={
                    x: batch_xs,
                    y: batch_ys, 
                })
        
                if step % 20 == 0:
                    print("Accuracy is :", sess.run( 
                        accuracy, feed_dict={
                            x: batch_xs,
                            y: batch_ys,
                        }))

                xs[0] = xs[1]
                ys[0] = ys[1]
                xs[1] = sess.run(
                    accuracy, feed_dict={
                        x: batch_xs,
                        y: batch_ys
                    }
                )
                ys[1] = step
                plt.plot(ys, xs)
                plt.pause(0.01)
                step = step + 1

            test_data = mnist.test.images[:128].reshape((
                -1, n_steps,
                n_inputs
                ))
            test_label = mnist.test.labels[:128]
    
            Show_Result['text'] = "Testing Accuracy :", sess.run(
                accuracy, feed_dict={
                    x: test_data,
                    y: test_label
                    }
            ) 
            save_path = saver.save(
                sess,
                path+"/Classification"+currtime+".ckpt"
                )
            b_OpenSavedRnn['state'] = "normal"
            plt.savefig(path+"/Classification"+currtime+".png")
            Show_location["text"] = "save the path sucessed to"+save_path
            sess.close()
        # t_Rnn.stop()


# The stop of Threading ################################################
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    if res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
 

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)
#########################################################################


def OnlyNumber(content):
    if content.isdigit() or content == "":
        return True
    else:
        return False
   

def Setting():
    path = askdirectory()
    b_Stop['state'] = "normal"
    training_iters = int(e_training.get())
    batch_size = int(e_batch.get())
    n_inputs = int(e_inputs.get())
    n_steps = int(e_steps.get())
    n_classes = int(e_classes.get())
    tr = Lstm()
    currtime = tr.currtime
    """
    tr.Lstm_RNN(
        training_iters,
        batch_size,
        n_inputs,
        n_steps,
        n_classes,
        path,
        currtime
    )
    """
    t_Rnn = threading.Thread(target=tr.Lstm_RNN, args=(
        training_iters,
        batch_size,
        n_inputs,
        n_steps,
        n_classes,
        path,
        currtime
    ))
    
    t_Rnn.start()
    t_update = threading.Thread(target=tr.UpdatePic, args=(path, currtime))
    t_update.start()


def SavePic():
    return 0                         


def Stop():
    t_Stop = threading.currentThread()
    t_Stop.do_run = False
    t_Stop.join()


def OpenValidImage():
    loadImagepath = tk.filedialog.askopenfilename()
    loadPara_path = loadRnnpath.replace('.ckpt.index', '.txt')
    loadPara = open(loadPara_path, "r")
    training_iters3 = int(loadPara.readline())
    n_inputs3 = int(loadPara.readline())
    n_steps3 = int(loadPara.readline())
    batch_size3 = int(loadPara.readline())
    n_classes3 = int(loadPara.readline())
    print(training_iters3)
    ld = Load_Lstm()
    ValidResult = ld.Loading_Rnn_Lstm(
        training_iters3,
        n_inputs3,
        n_steps3,
        batch_size3,
        n_classes3,
        loadRnnpath,
        loadImagepath
    )
    Show_Result2['text'] = "Recognized Image is", ValidResult


def OpenSavedRnn():
    global loadRnnpath
    loadRnnpath = tk.filedialog.askopenfilename()


OnlyNumberR = window.register(OnlyNumber)

# Left Label ############################
logotest = Image.open("Save_NN/logotrans5.png")
logo = ImageTk.PhotoImage(logotest)
l_logo = tk.Label(window, image=logo,).place(
    x=0,
    y=0,
    anchor='nw'
)
l_training = tk.Label(window, text='Iterations').place(
    x=50,
    y=110,
    anchor='nw'
)
e_training = tk.Entry(
    window,
    validate='key',
    validatecommand=(OnlyNumberR, '%P'),
    show=None
    )
e_training.place(
    x=150,
    y=110,
    anchor='nw'
    )
l_inputs = tk.Label(window, text='inputs').place(
    x=50,
    y=150,
    anchor='nw'
)
e_inputs = tk.Entry(
    window,
    validate='key',
    validatecommand=(OnlyNumberR, '%P'),
    show=None
    )
e_inputs.place(
    x=150,
    y=150,
    anchor='nw'
    )
l_steps = tk.Label(window, text='steps').place(
    x=50,
    y=190,
    anchor='nw'
)
e_steps = tk.Entry(
    window,
    validate='key',
    validatecommand=(OnlyNumberR, '%P'),
    show=None)
e_steps.place(
    x=150,
    y=190,
    anchor='nw'
    )
l_batch = tk.Label(window, text='batch size').place(
    x=50,
    y=230,
    anchor='nw'
)
e_batch = tk.Entry(
    window,
    validate='key',
    validatecommand=(OnlyNumberR, '%P'),
    show=None)
e_batch.place(
    x=150,
    y=230,
    anchor='nw'
    )

l_classes = tk.Label(window, text='classes').place(
    x=50,
    y=270,
    anchor='nw'
)
e_classes = tk.Entry(
    window,
    validate='key',
    validatecommand=(OnlyNumberR, '%P'),
    show=None)
e_classes.place(
    x=150,
    y=270,
    anchor='nw'
    )

# Button #################################
b_Setting = tk.Button(
    window, text='go',
    width=20,
    height=4,
    command=Setting
    )
b_Setting.place(
    x=360,
    y=100,
    anchor='nw'
    )

b_Stop = tk.Button(
    window, text='stop',
    width=20,
    height=4,
    state="disabled",
    command=Stop
 )
b_Stop.place(
    x=360,
    y=220,
    anchor='nw'
 )

b_SavePic = tk.Button(
    window, text='Save the Picture',
    width=20,
    height=2,
    command=SavePic
 )
b_SavePic.place(
     x=700,
     y=550,
     anchor='nw'
 )

b_OpenSavedRnn = tk.Button(
    window, text='Open the Saved Rnn',
    width=20,
    height=4,
    state="normal",
    command=OpenSavedRnn
 )
b_OpenSavedRnn.place(
     x=360,
     y=400,
     anchor='nw'
 )
b_OpenValidImage = tk.Button(
    window, text="Upload and Validtion",
    width=20,
    height=4,
    state="normal",
    command=OpenValidImage
)
b_OpenValidImage.place(
    x=360,
    y=500,
    anchor='nw'
)
# Right Image ##############################
Show_Image = tk.Label(window, width=800, height=500)
Show_Image.place(x=530, y=10, anchor='nw')
# Menu #####################################

# Left Label Result #######################
Show_Result = tk.Label(window, text='123', width=40, height=3)
Show_Result.place(x=50, y=310, anchor='nw')
Show_location = tk.Label(window, width=70, height=3)
Show_location.place(x=50, y=350, anchor='nw')
Show_Result2 = tk.Label(window, text='1234', width=20, height=3)
Show_Result2.place(x=50, y=400, anchor='nw')
# Copyright ###################
CopyR = tk.Label(window, text='Copyright@ ', width=20, height=4)
CopyR.place(x=10, y=550, anchor='nw')
plt.show()
window.mainloop()