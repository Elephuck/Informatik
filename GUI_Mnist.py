import tkinter as tk
from tkinter.filedialog import askdirectory
import threading
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# this is data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# windows
window = tk.Tk()
window.title('LSTM')
window.geometry('1300x600')
training_iters = 40000
batch_size = 128
# window2.withdraw()
n_inputs = 28
n_steps = 28
n_classes = 10
training_iters2 = 0
n_inputs2 = 0
n_steps2 = 0
batch_size2 = 0
n_classes2 = 0
global pathRNN
global StopTheRnn
StopTheRnn = 0
pathRNN = "abc"
'''
Thread reload run()
'''


'''
This section is about Entry and Label
'''


def OnlyNumber(content):
    if content.isdigit() or content == "":
        return True
    else:
        return False


OnlyNumberR = window.register(OnlyNumber)

# Left Label
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

# Left Label Result
Show_Result = tk.Label(window, text='123', width=40, height=10)
Show_Result.place(x=50, y=310, anchor='nw')
# Copyright
CopyR = tk.Label(window, text='Copyright@ ', width=20, height=4)
CopyR.place(x=10, y=550, anchor='nw')


'''
Form here is RNN 
'''


# def RNN
def RNN(X, weights, biases, n_inputs, n_steps, n_hidden_units):
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
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
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


def RNN_load(X, weights, biases, n_inputs, n_steps, n_hidden_units):
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
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
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


# Event
Stop_RNN = threading.Event()


# Button Funktion
def Lstm_RNN(training_iters, batch_size, n_inputs, n_steps, n_classes):
    print('traing', training_iters)
    print('batch', batch_size)
    print('inputs', n_inputs)
    print('steps', n_steps)
    print('classes', n_classes)
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
    
    pred = RNN(x, weights, biases, n_inputs, n_steps, n_hidden_units)
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
            if(StopTheRnn == 1):
                    plt.savefig("Save_NN/Classification.png")
                    print("stop RNn")
                    break
            else:
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
            })
        save_path = saver.save(sess, "Save_NN/Classification.ckpt")
        b_OpenSavedRnn['state'] = "normal"
    plt.savefig("Save_NN/Classification.png")
    print("save the path sucessed to", save_path)
    t_update.start()
    sess.close()


def Loading_Rnn_Lstm(
      training_iters,
      batch_size,
      n_inputs,
      n_steps,
      n_classes,
      loadpath,
      loadImagepath):
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
    weights = {
 
      'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),

      'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
      }
    biases = {

      'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
      'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
      }
    
    pred = RNN_load(x, weights, biases, n_inputs, n_steps, n_hidden_units)
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, loadpath+"/Classification.ckpt")
            load_Result = sess.run(
                tf.argmax(pred, 1),
                feed_dict={x: fs}
            )
    return load_Result
    sess.close()
           

# Right Label for window1
Show_Image = tk.Label(window, width=800, height=500)
Show_Image.place(x=530, y=10, anchor='nw')


def Run_image(m):
    Show_Image['image'] = m
 

# image Label
imgtest = Image.open("Save_NN/IMG_9205.jpg")
phototest = ImageTk.PhotoImage(imgtest)
Show_Image['image'] = phototest


# Thread Function for Update
def UpdatePic():
    while(1):
        if(os.path.exists("Save_NN/Classification.png")):
            time.sleep(1)
            img = Image.open("Save_NN/Classification.png")
            photo = ImageTk.PhotoImage(img)
            Run_image(photo)
            time.sleep(3)
        else:
            print("no image")
            time.sleep(3)


t_update = threading.Thread(target=UpdatePic)


# Function of Button
def Setting():
    b_Setting['state'] = "disabled"
    b_Stop['state'] = "normal"
    training_iters = int(e_training.get())
    batch_size = int(e_batch.get())
    n_inputs = int(e_inputs.get())
    n_steps = int(e_steps.get())
    n_classes = int(e_classes.get())
    doc = open('Save_NN/Parameter.txt', 'w+')
    print(training_iters, file=doc)
    print(batch_size, file=doc)
    print(n_inputs, file=doc)
    print(n_steps, file=doc)
    print(n_classes, file=doc)
    t_Rnn = threading.Thread(target=Lstm_RNN, args=(
        training_iters,
        batch_size,
        n_inputs,
        n_steps,
        n_classes
    ))
    t_Rnn.start()


def Stop():
    # StopTheRnn = 1
    window.destroy()


def SavePic():
    path = askdirectory()
    pathPic = path
    plt.savefig(pathPic+"/Classfication.png")
    print("save the path sucessed to", pathPic)


def OpenSavedRnn():
    os.system("start explorer D:\python\Save_NN")


def New_training():
    window.update()


def Load_training():
    window.withdraw()
    global window2
    window2 = tk.Toplevel()
    window2.title('Loading the Lstm')
    window2.geometry('600x600')
    l2_logo = tk.Label(window2, image=logo)
    l2_logo.place(
        x=0,
        y=0,
        anchor='nw'
    )

    # left Label for windows 2
    b_loadParameter = tk.Button(
        window2, text='load the saved RNN',
        width=20,
        height=4,
        command=B_LoadRnnPara
        )
    b_loadParameter.place(
        x=360,
        y=100,
        anchor='nw'
    )
    global b2_uploadImage
    b2_uploadImage = tk.Button(
        window2, 
        text='Upload the Image',
        width=20,
        height=4,
        command=B_LoadImage,
        state='disabled'
        )
    b2_uploadImage.place(
        x=360,
        y=220,
        anchor='nw'
    )
    l2_training = tk.Label(window2, text='Iterations')
    l2_training.place(
        x=50,
        y=110,
        anchor='nw'
    )
    l2_inputs = tk.Label(window2, text='inputs')
    l2_inputs.place(
        x=50,
        y=150,
        anchor='nw'
    )
    l2_steps = tk.Label(window2, text='steps')
    l2_steps.place(
        x=50,
        y=190,
        anchor='nw'
    )
    l2_batch = tk.Label(window2, text='batch size')
    l2_batch.place(
        x=50,
        y=230,
        anchor='nw'
    )
    l2_classes = tk.Label(window2, text='classes')
    l2_classes.place(
        x=50,
        y=270,
        anchor='nw'
    )
    # Left Label Result for Windows 2
    global Show_Result2
    Show_Result2 = tk.Label(window2, text='1234', width=40, height=10)
    Show_Result2.place(x=50, y=310, anchor='nw')
    # Right Label for Window 2


def Exit_training():
    sys.exit(0)


def B_LoadRnnPara():
    global load_Rnnpath
    load_Rnnpath = tk.filedialog.askopenfilename()
    loadpath = os.path.dirname(load_Rnnpath)
    loadf = open(loadpath+"/Parameter.txt", "r")
    '''
    t2_Rnn = threading.Thread(target=UpdatePic2, args=loadpathpic)
    loadpic = Image.open(loadpath+"/Classification.png")
    loadphoto = ImageTk.PhotoImage(loadpic)
    load_the_pic(loadphoto)
    '''
    training_iters2 = int(loadf.readline())
    n_inputs2 = int(loadf.readline())
    n_steps2 = int(loadf.readline())
    batch_size2 = int(loadf.readline())
    n_classes2 = int(loadf.readline())
    print(training_iters2)
    print(n_inputs2)
    print(n_steps2)
    print(batch_size2)
    print(n_classes2)
    l2_trainingp = tk.Label(window2, text=training_iters2)
    l2_trainingp.place(
        x=150,
        y=110,
        anchor='nw'
    )
    l2_inputsp = tk.Label(window2, text=n_inputs2)
    l2_inputsp.place(
        x=150,
        y=150,
        anchor='nw'
    )
    l2_stepsp = tk.Label(window2, text=n_steps2)
    l2_stepsp.place(
        x=150,
        y=190,
        anchor='nw'
    )
    l2_batchp = tk.Label(window2, text=batch_size2)
    l2_batchp.place(
        x=150,
        y=230,
        anchor='nw'
    )
    l2_classesp = tk.Label(window2, text=n_classes2)
    l2_classesp.place(
        x=150,
        y=270,
        anchor='nw'
    ) 
    b2_uploadImage['state'] = 'normal'


def B_LoadImage():
    loadImage_path = tk.filedialog.askopenfilename()
    loadRNN_path = loadImage_path
    loadpath = os.path.dirname(loadRNN_path)
    loadf = open(loadpath+"/Parameter.txt", "r")
    training_iters3 = int(loadf.readline())
    n_inputs3 = int(loadf.readline())
    n_steps3 = int(loadf.readline())
    batch_size3 = int(loadf.readline())
    n_classes3 = int(loadf.readline())
    load_Result = Loading_Rnn_Lstm(
        training_iters3,
        n_inputs3,
        n_steps3,
        batch_size3,
        n_classes3,
        loadpath,
        loadImage_path
        )
    Show_Result2['text'] = "Recognized Image is", load_Result
    

# Button
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
    window, text='Open the Saved RNN',
    width=20,
    height=2,
    state="disabled",
    command=OpenSavedRnn
 )
b_OpenSavedRnn.place(
     x=1000,
     y=550,
     anchor='nw'
 )

# Menu
menubar1 = tk.Menu(window)
Mainmenu = tk.Menu(menubar1, tearoff=0)
menubar1.add_cascade(label="Menu", menu=Mainmenu)
Mainmenu.add_command(label="New Training", command=New_training)
Mainmenu.add_command(label="Load The Saved Training", command=Load_training)
Mainmenu.add_command(label="Exit", command=Exit_training)
window.config(menu=menubar1)

plt.show()
window.mainloop()