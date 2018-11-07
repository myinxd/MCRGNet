# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>

"""
A class to define the convolutional autoencoder (CAE) by tensorflow.

Reference
=========
[1] tensorflow_tutorials
    https://github.com/pkmital/tensorflow_tutorials/blob/master/
    python/07_autoencoder.py

Update
======
[2017-09-04] Add methods to fine-tuning the network.
[2017-09-04] Change padding and stride strtegy to custom providing.
[2017-09-04] Remove pooling staffs

[2017-09-07] Add validation
[2017-09-07] Add accuracy evalutation of CNN, i.e., fine-tuning
[2017-09-07] Add draw_training_curve()

[2017-09-14] Variable space to fix the checkpoint, and solve restore problem
[2018-05-03] Add adjustable learning rate
[2018-08-24] Fixed and adjusted parameters
"""

import tensorflow as tf
import numpy as np
import time

class ConvAE():
    """
    A convolutional autoencoder (CAE) constructor

    inputs
    ======
    input_shape: np.ndarray
        Shape of the input of the network = (d,r,c).
        d: dimensions,
        r: rows,
        c: cols
    kernel_size: list
        Shape of the kernels in each ConvLayer
    Kernel_num: list
        Number of kernels in each ConvLayer
    fc_nodes: list
        The dense layers after the fully connected layer
        of the last ConvLayer or pooling layer
    encode_nodes: int
        Number of nodes in the encode layer
    padding: tuple (pad_en, pad_de)
        Padding modes for encode and decode blocks,
        default as ('SAME','SAME')
    stride: 2D tuple
        Stride step, default as (2, 2)
    numclass: int
        Number of classes at the sofrmax layer
    valrate: float
        Validation rate, range from 0.0 to 0.1, default as 0.3
    sess: tf.Session
        The restored session of pretrained, default as None
    name: str
        Name of the saved pretrained network

    Methods
    =======
    gen_BatchIterator: Generate a iterator for batch training.
    cae_build: Build the CAE network.
    gen_cost: Generate the CAE cost function.
    gen_optimizer: Generate the ADAM optimizer.
    gen_validation: Get training and validation subsets with provided rate.
    cae_train: Train the CAE network.
    cae_test: Test the CAE network.
    cae_encode: Get the feature vector of the input image.
    cae_decode: Reconstruct from a feature vector.
    cae_save: Save the CAE network.
    gen_cnn_BatchIterator: Generate a iterator for cnn batch training.
    cnn_build: Build the CNN network.
    gen_cnn_cost: Generate the CNN cost function.
    gen_cnn_optimizer: Generate the CNN optimizer.
    cnn_train: Train the CNN network.
    cnn_predict: Predict or classify a RG image.
    gen_accuracy: Generate the accuracy tensor.
    cnn_accuracy: Test The CNN network.
    cnn_save: Save the CNN network.
    """

    def __init__(self, input_shape,
                 kernel_size=[3,3,3], kernel_num=[10,10,10],
                 fc_nodes=[64],
                 encode_nodes=10,
                 padding = ('SAME','SAME'),
                 stride = (2, 2),
                 numclass = 2,
                 valrate = 0.3,
                 sess = None,
                 name = None,
                 ):
        """
        The initializer
        """
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.fc_nodes = fc_nodes
        self.encode_nodes = encode_nodes
        self.pad_en = padding[0]
        self.pad_de = padding[1]
        self.stride = stride
        self.numclass = numclass # used for fine tuning, maybe need reconsidered.
        self.y_ = tf.placeholder(tf.float32, shape=[None, numclass], name="cnn-labels") # placeholder of y
        self.l_cnn = tf.placeholder(tf.float32, shape=[None, numclass], name="l_cnn") # cnn
        self.l_en = tf.placeholder(tf.float32, shape=[None, self.encode_nodes], name="l_en")
        # Input layer
        in_depth = self.input_shape[3]
        in_row = self.input_shape[1]
        in_col = self.input_shape[2]
        self.l_in = tf.placeholder(
            tf.float32,
            [None,in_row,in_col,in_depth],
            name='l_in')
        # decoder layer
        self.l_de = tf.placeholder(tf.float32,
                                   [None,in_row,in_col,in_depth], name='l_de')
        self.droprate = tf.placeholder(tf.float32, name="droprate")
        self.valrate = 0.2
        self.get_learning_rate()
        # sess for reload
        self.sess = sess
        if self.sess is not None:
            try:
                self.l_in = sess.graph.get_tensor_by_name(name['l_in'])
                self.l_en = sess.graph.get_tensor_by_name(name['l_en'])
                self.l_de = sess.graph.get_tensor_by_name(name['l_de'])
            except:
                print("Some thing wrong when load the pretrained network...")

    def gen_BatchIterator(self, data, batch_size=100, shuffle=True):
        """
        Return the next 'batch_size' examples
        from the X_in dataset

        Reference
        =========
        [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
        Input
        =====
        data: 4d np.ndarray
            The samples to be batched
        batch_size: int
            Size of a single batch.
        shuffle: bool
            Whether shuffle the indices.

        Output
        ======
        Yield a batch generator
        """
        if shuffle:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
        for start_idx in range(0, len(data) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield data[excerpt]


    def get_learning_rate(self):
        """Get the exponentially decreased learning rate."""
        self.init_lr = tf.placeholder(tf.float32, name="init_lr")
        self.global_step = tf.placeholder(tf.float32, name="global_step")
        self.decay_step = tf.placeholder(tf.float32, name="decay_step")
        self.decay_rate = tf.placeholder(tf.float32, name="decay_rate")
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.init_lr ,
            global_step=self.global_step,
            decay_steps=self.decay_step,
            decay_rate=self.decay_rate,
            staircase=False,
            name="learning_rate")


    # Pre-training part
    def cae_build(self):
        """Construct the network, including ConvLayers and FC layers."""

        # Useful methods
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        # Init
        self.encoder = []
        self.shapes_en = []

        # Encoder layers
        current_input = self.l_in
        for i, depth_output in enumerate(self.kernel_num):
            depth_input = current_input.get_shape().as_list()[3]
            self.shapes_en.append(current_input.get_shape().as_list())
            W_name = "Conv_En_W{0}".format(i)
            b_name = "Conv_En_b{0}".format(i)
            W = weight_variable(
                shape=[self.kernel_size[i],
                       self.kernel_size[i],
                       depth_input,
                       depth_output],
                name=W_name)
            b = bias_variable(shape=[depth_output], name=b_name)
            self.encoder.append(W) # share
            output = tf.add(
                tf.nn.conv2d(current_input,
                             W, strides=[1,self.stride[0],self.stride[1],1],
                             padding=self.pad_en), b)
            output = tf.nn.relu(output)
            output = tf.nn.dropout(output, self.droprate)
            current_input = output

        if self.encode_nodes is not None:
            # Dense layer
            shape_conv = current_input.get_shape().as_list()
            depth_dense = shape_conv[1] * shape_conv[2] * shape_conv[3]
            l_en_dense = tf.reshape(current_input, [-1, depth_dense])

            # fully connected
            depth_input = depth_dense
            current_input = l_en_dense
            self.en_fc = [] # save and share weights of the fc layers
            self.fc_depth=[] # depth_input of the encoder
            for i, depth_output in enumerate(self.fc_nodes):
                self.fc_depth.append(depth_input)
                W_name = "FC_En_W{0}".format(i)
                b_name = "FC_En_b{0}".format(i)
                W = weight_variable(shape=[depth_input, depth_output], name=W_name)
                b = bias_variable(shape=[depth_output], name=b_name)
                self.en_fc.append(W) # share weight
                output = tf.nn.relu(tf.matmul(current_input, W) + b)
                # dropout
                output = tf.nn.dropout(output, self.droprate)
                current_input = output
                depth_input = depth_output

            # encode layer
            W_en = weight_variable(shape=[depth_input, self.encode_nodes], name="En_W")
            b_en = bias_variable(shape=[self.encode_nodes], name="En_b")
            output = tf.nn.relu(tf.matmul(current_input, W_en) + b_en)
            current_input = output
            self.l_en = current_input

            # decoder layers
            W_de = tf.transpose(W_en)
            if len(self.fc_nodes):
                depth_output = self.fc_nodes[-1]
            else:
                depth_output = depth_dense
            b_de = bias_variable(shape=[depth_output], name="De_b")
            output = tf.nn.relu(tf.matmul(current_input, W_de) + b_de)
            current_input = output

            # fc layers
            for i in range(len(self.fc_nodes)-1, -1, -1):
                depth_output = self.fc_depth[i]
                b_name = "FC_De_b{0}".format(i)
                W = tf.transpose(self.en_fc[i])
                b = bias_variable(shape=[depth_output], name=b_name)
                output = tf.nn.relu(tf.matmul(current_input, W) + b)
                # dropout
                output = tf.nn.dropout(output, self.droprate)
                current_input = output

            # Dense layer
            # reshape
            current_input = tf.reshape(current_input,
                                    [-1,
                                     shape_conv[1],
                                     shape_conv[2],
                                     shape_conv[3]])
        else:
            self.l_en = current_input

        self.decoder = self.encoder.copy()
        self.decoder.reverse()
        self.shapes_de = self.shapes_en.copy()
        self.shapes_de.reverse()

        for i, shape in enumerate(self.shapes_de):
            W = self.decoder[i]
            b_name = "Conv_De_b{0}".format(i)
            b = bias_variable(shape=(W.get_shape().as_list()[2],), name=b_name)
            output = tf.add(tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(self.l_in)[0], shape[1], shape[2], shape[3]]),
                strides = [1,self.stride[0],self.stride[1],1], padding=self.pad_de),
                b)
            output = tf.nn.relu(output)
            output = tf.nn.dropout(output, self.droprate)
            current_input = output

        # Decoder output
        self.l_de = current_input

    def gen_cost(self):
        """Generate the cost, usually be the mean square error."""
        # cost function
        self.cost = tf.reduce_mean(tf.square(self.l_de-self.l_in))

    def gen_optimizer(self, learning_rate=0.01):
        """Generate the optimizer to minimize the cose"""
        self.gen_cost()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name="cae-optimizer").minimize(self.cost)

    def gen_validation(self, data, label=None):
        """Separate the dataset into training and validation subsets.

        inputs
        ======
        data: np.ndarray
            The input data, 4D matrix
        label: np.ndarray or list, opt
            The labels w.r.t. input data, optional

        outputs
        =======
        data_train: {"data": , "label": }
        data_val: {"data":, "label":}
        """
        if label is None:
            label_train = None
            label_val = None
            idx = np.random.permutation(len(data))
            num_val = int(len(data)*self.valrate)
            data_val = {"data": data[idx[0:num_val],:,:,:],
                        "label": label_val}
            # train
            data_train = {"data": data[idx[num_val:],:,:,:],
                          "label": label_train}
        else:
            # Training and validation
            idx = np.random.permutation(len(data))
            num_val = int(len(data)*self.valrate)
            data_val = {"data": data[idx[0:num_val],:,:,:],
                        "label": label[idx[0:num_val],:]}
            # train
            data_train = {"data": data[idx[num_val:],:,:,:],
                          "label": label[idx[num_val:],:]}

        return data_train,data_val

    def cae_train(self, data, num_epochs=20,
                  learning_rate=0.01, derate=1/16, deepoch=20,
                  batch_size=100, droprate=0.5):
        """Train the net"""
        timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training parameters\n" % (timestamp))
        print("[%s] Epochs: %d\tLearning rate: %.2f\n" % (timestamp, num_epochs, learning_rate))
        print("[%s] Batch size: %d\tDrop rate: %.2f\n" % (timestamp, batch_size, droprate))
        self.gen_optimizer(learning_rate = learning_rate)
        init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init_op)
        # save loss for drawing
        self.cae_trainloss = []
        self.cae_valloss = []

        decay_rate = 0.95
        # generate training and validation samples
        data_train,data_val = self.gen_validation(data=data, label=None)
        numbatch_trn = len(data_train["data"]) // batch_size
        numbatch_val = len(data_val["data"]) // batch_size
        for epoch in range(num_epochs):
            # update learning rate
            lr_dict = {
                self.init_lr: learning_rate,
                self.global_step: epoch,
                self.decay_step: batch_size,
                self.decay_rate: decay_rate}
            cost_trn = 0
            cost_val = 0
            for batch in self.gen_BatchIterator(data=data_train["data"], batch_size=batch_size):
                trn_dict = {self.l_in: batch, self.droprate: droprate}
                trn_dict.update(lr_dict)
                self.sess.run(self.optimizer, feed_dict=trn_dict)
                cost_trn += self.sess.run(self.cost,
                               feed_dict={self.l_in: batch, self.droprate: 0.0})

            for batch in self.gen_BatchIterator(data=data_val["data"], batch_size=batch_size):
                cost_val += self.sess.run(self.cost, feed_dict={self.l_in: batch, self.droprate: 0.0})

            timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
            print("[%s] Epoch: %03d\tTraining loss: %.6f\tValidation loss: %.6f" %
                (timestamp, epoch+1, cost_trn/numbatch_trn, cost_val/numbatch_val))
            self.cae_trainloss.append(cost_trn/numbatch_trn)
            self.cae_valloss.append(cost_val/numbatch_val)

    def cae_test(self, img):
        """Test the trained network,

        Input
        =====
        img: np.ndarray
            The images to be test.

        Output
        ======
        img_pre: np.ndarray
            The predicted images.
        """
        # params
        depth = self.input_shape[3]
        rows = self.input_shape[1]
        cols = self.input_shape[2]
        # Reshape the images
        shapes = img.shape
        if len(shapes) == 2:
            if shapes[0] != rows or shapes[1] != cols:
                print('The shape of the test images do not match the network.')
                return None
            img_te = img.reshape(1,rows,cols,depth)
        elif len(shapes) == 3:
            if shapes[0] != rows or shapes[1] != cols or shapes[2] != depth:
                print('The shape of the test images do not match the network.')
                return None
            img_te = img.reshape(1,rows,cols,depth)
        elif len(shapes) == 4:
            if shapes[1] != rows or shapes[2] != cols or shapes[3] != depth :
                print('The shape of the test images do not match the network.')
                return None
            img_te = img.reshape(shapes[0],rows,cols,depth)

        # generate predicted images
        if hasattr(self, 'sess'):
            img_pre = self.sess.run(self.l_de, feed_dict={self.l_in: img_te, self.droprate: 0.0})
        else:
            print("You should train the network.")
            return None

        return img_pre

    def cae_encode(self, img):
        """Test the trained network,

        Input
        =====
        img: np.ndarray
            The images to be test.

        Output
        ======
        code: np.ndarray
            The encoded codes.
        """
        # params
        depth = self.input_shape[3]
        rows = self.input_shape[1]
        cols = self.input_shape[2]
        # Reshape the images
        shapes = img.shape
        if len(shapes) == 2:
            if shapes[0] != rows or shapes[1] != cols:
                print('The shape of the test images do not match the network.')
                return None
            img_te = img.reshape(1,rows,cols,depth)
        elif len(shapes) == 3:
            if shapes[0] != rows or shapes[1] != cols or shapes[2] != depth:
                print('The shape of the test images do not match the network.')
                return None
            img_te = img.reshape(1,rows,cols,depth)
        elif len(shapes) == 4:
            if shapes[1] != rows or shapes[2] != cols or shapes[3] != depth :
                print('The shape of the test images do not match the network.')
                return None
            img_te = img.reshape(shapes[0],rows,cols,depth)

        # generate predicted images
        if hasattr(self, 'sess'):
            code = self.sess.run(self.l_en, feed_dict={self.l_in: img_te, self.droprate: 0.0})
        else:
            print("You should train the network.")
            return None

        return code

    def cae_decode(self, code):
        """Generate agn images with provided codes.

        Input
        =====
        code: np.ndarray
            The code to be decoded.

        Output
        ======
        img_de: np.ndarray
            The decoded and reconstructed images.
        """
        # Compare code length
        code_len = self.l_en.get_shape()[1]
        if code.shape[1] != code_len:
            print("The length of provided codes should be equal to the network's")
            return None
        else:
            # decoding
            if hasattr(self, 'sess'):
                l_in_shape = self.l_in.get_shape().as_list()
                l_in_shape[0] = code.shape[0]
                p_in = np.zeros(l_in_shape) # pseudo input
                img_de = self.sess.run(self.l_de,
                                       feed_dict={self.l_in: p_in,
                                                  self.l_en: code,
                                                  self.droprate: 0.0})
            else:
                print("You should train the network firstly.")
                return None

        return img_de

    def cae_save(self, namepath, netpath):
        """Save the net"""
        import pickle
        import sys
        sys.setrecursionlimit(1000000)

        # save the names
        savedict = {'l_in': self.l_in.name,
                    'l_en': self.l_en.name,
                    'l_de': self.l_de.name,
                    'l_cnn': self.l_cnn.name,
                    'y_':self.y_.name,
                    'cnn_cost':self.cnn_cost.name,
                    'droprate': self.droprate.name,
                    'netpath': netpath}
        with open(namepath, 'wb') as fp:
            pickle.dump(savedict, fp)

        # save the net
        saver = tf.train.Saver()
        saver.save(self.sess, netpath)

    ##################
    #Fine-tuning part#
    ##################
    def gen_cnn_BatchIterator(self, data, label, batch_size=100, shuffle=True):
        """
        Return the next 'batch_size' examples
        from the X_in dataset

        Reference
        =========
        [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
        Input
        =====
        data: 4d np.ndarray
            The samples to be batched
        label: np.ndarray
            The labels to be batched w.r.t. data
        batch_size: int
            Size of a single batch.
        shuffle: bool
            Whether shuffle the indices.

        Output
        ======
        Yield a batch generator
        """
        if shuffle:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
        else:
            indices = np.arange(len(data))
        data = data[indices]
        label = label[indices]
        numbatch = len(indices) // batch_size
        return data,label,numbatch

    def cnn_build(self):
        """Build the cnn for fine-tuning, just add a softmax layer after the encoder layer.
           Since the weights are shared between encoder and decoder, the fine-tuned weights
           can be reused.
        """
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)
        # add a softmax layer
        W_soft = weight_variable([self.encode_nodes, self.numclass], name="cnn_W")
        b_soft = bias_variable([self.numclass], name="cnn_b")

        self.l_cnn = tf.nn.softmax(tf.matmul(self.l_en, W_soft) + b_soft)
        # generate the optimizer
        self.gen_cnn_optimizer()
        # generate the accuracy estimator
        self.gen_accuracy()

    def gen_cnn_cost(self):
        """Generate the cost function for the added softmax layer at the end of encoder."""
        # cost function
        self.cnn_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.l_cnn))

    def gen_cnn_optimizer(self):
        """Generate the optimizer for the fine-tuning cnn.

        Reference
        =========
        [1] Tensorflow: No gradients provided for any variable
            https://stackoverflow.com/questions/38778760/tensorflow-no-gradients-provided-for-any-variable
        """
        self.gen_cnn_cost()
        self.cnn_optimizer = tf.train.AdamOptimizer(self.learning_rate,name='cnn-optimizer').minimize(self.cnn_cost)

    def cnn_train(self, data, label, num_epochs=20, learning_rate=0.0005, batch_size=100, droprate=0.5):
        """Train the net"""
        timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training parameters\n" % (timestamp))
        print("[%s] Epochs: %d\tLearning rate: %.2f\n" % (timestamp, num_epochs, learning_rate))
        print("[%s] Batch size: %d\tDrop rate: %.2f\n" % (timestamp, batch_size, droprate))
        if self.sess is None:
            init_op = tf.global_variables_initializer()
            self.sess = tf.InteractiveSession()
            self.sess.run(init_op)

        # save loss for drawing
        self.cnn_trainloss = []
        self.cnn_valloss = []
        self.cnn_trainacc = []
        self.cnn_valacc = []

        decay_rate = 0.95
        # generate training and validation samples
        data_train,data_val = self.gen_validation(data=data, label=label)
        for epoch in range(num_epochs):
            cost = 0
            acc = 0
            data_batch,label_batch,numbatch = self.gen_cnn_BatchIterator(
                data=data_train["data"], label=data_train["label"], batch_size=batch_size)
            # update learning rate
            lr_dict = {
                self.init_lr: learning_rate,
                self.global_step: epoch,
                self.decay_step: batch_size,
                self.decay_rate: decay_rate}
            for i in range(numbatch):
                batch = data_batch[i*batch_size:(i+1)*batch_size,:,:,:]
                batch_label = label_batch[i*batch_size:(i+1)*batch_size,:]
                trn_dict = {self.l_in: batch, self.y_: batch_label, self.droprate: droprate}
                trn_dict.update(lr_dict)
                # print(batch_label.shape)
                self.sess.run(self.cnn_optimizer,
                              feed_dict=trn_dict)
                cost += self.sess.run(
                    self.cnn_cost,
                    feed_dict={self.l_in: batch, self.y_: batch_label, self.droprate: 0.0})
                acc += self.cnn_accuracy(img=batch, label=batch_label)
            # validation
            valloss = self.sess.run(
                self.cnn_cost,
                feed_dict={self.l_in: data_val["data"],
                           self.y_: data_val["label"],
                           self.droprate: 0.0})
            valacc = self.cnn_accuracy(img=data_val["data"], label=data_val["label"])
            timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
            print("[%s] Epoch: %03d Trn loss: %.6f Trn acc: %.3f Val loss: %.6f Val acc: %.3f" %
                (timestamp, epoch+1, cost/numbatch, acc/numbatch, valloss, valacc))
            self.cnn_trainloss.append(cost/numbatch)
            self.cnn_valloss.append(valloss)
            self.cnn_trainacc.append(acc/numbatch)
            self.cnn_valacc.append(valacc)

    def cnn_predict(self, img):
        """Predict types of the sample with the
           trained fine-tuned network.

        Input
        =====
        img: np.ndarray
            The images to be test.

        Output
        ======
        label_pre: np.ndarray
            The predicted label.
        """
        # params
        depth = self.input_shape[3]
        rows = self.input_shape[1]
        cols = self.input_shape[2]
        # Reshape the images
        shapes = img.shape
        if len(shapes) == 2:
            if shapes[0] != rows or shapes[1] != cols:
                print('The shape of the test images do not match the network.')
                return None
            img_te = img.reshape(1,rows,cols,depth)
        elif len(shapes) == 3:
            if shapes[0] != rows or shapes[1] != cols or shapes[2] != depth:
                print('The shape of the test images do not match the network.')
                return None
            img_te = img.reshape(1,rows,cols,depth)
        elif len(shapes) == 4:
            if shapes[1] != rows or shapes[2] != cols or shapes[3] != depth :
                print('The shape of the test images do not match the network.')
                return None
            img_te = img.reshape(shapes[0],rows,cols,depth)

        # generate predicted images
        if hasattr(self, 'sess'):
            label_pre = tf.argmax(self.sess.run(self.l_cnn, feed_dict={self.l_in: img_te, self.droprate: 0.0}))
        else:
            print("You should train the network.")
            return None

        return label_pre.eval()

    def gen_accuracy(self):
        """Generate the accuracy tensor"""
        self.correction_prediction = tf.equal(tf.argmax(self.l_cnn, 1), tf.argmax(self.y_,1), name="corr_pre")
        self.accuracy = tf.reduce_mean(tf.cast(self.correction_prediction, tf.float32), name="acc")

    def cnn_accuracy(self, img, label):
        """Calculate the classification accuracy of the cnn network.

        inputs
        ======
        img: np.ndarray
            The images to be classified
        label: np.ndarray, one hot
            The labels w.r.t. the images

        output
        ======
        acc: float
            Classification accuracy
        """
        return self.accuracy.eval(feed_dict={self.l_in: img, self.y_: label, self.droprate: 0.0})

    def cnn_save(self, namepath, netpath):
        """Save the net"""
        import pickle
        import sys
        sys.setrecursionlimit(1000000)

        # save the names
        savedict = {'l_in': self.l_in.name,
                    'l_cnn': self.l_cnn.name,
                    'y_':self.y_.name,
                    'droprate': self.droprate.name,
                    'netpath': netpath}
        with open(namepath, 'wb') as fp:
            pickle.dump(savedict, fp)

        # save the net
        saver = tf.train.Saver()
        saver.save(self.sess, netpath)
