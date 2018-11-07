# copyright (C) 2018 zxma_sjtu@qq.com

"""
The utils for construcing the convolutional auto-encoder (CAE) model.

"""

import os
import pickle
import numpy as np
import tensorflow as tf
import scipy.io as sio
from astropy.io import fits
from scipy.misc import imread
from skimage import transform
from astropy.stats import sigma_clip

def gen_sample(folder, ftype='jpg', savepath=None,
                crop_box=(200, 200), res_box=(50, 50), clipflag=False, clipparam=None):
    """
    Read the sample images and reshape to required structure

    input
    =====
    folder: str
        Name of the folder, i.e., the path
    ftype: str
        Type of the images, default as 'jpg'
    savepath: str
        Path to save the reshaped sample mat
        default as None
    crop_box: tuple
        Boxsize of the cropping of the center region
    res_box: tuple
        Scale of the resized image
    clipflag: booling
        The flag of sigma clipping, default as False
    clipparam: list
        Parameters of the sigma clipping, [sigma, iters]

    output
    ======
    sample_mat: np.ndarray
        The sample matrix
    """
    # Init
    if os.path.exists(folder):
        sample_list = os.listdir(folder)
    else:
        return

    sample_mat = np.zeros((len(sample_list),
                           res_box[0]*res_box[1]))

    def read_image(fpath,ftype):
        if ftype == 'fits':
            h = fits.open(fpath)
            img = h[0].data
        else:
            img = imread(name=fpath, flatten=True)
        return img

    # load images
    idx = 0
    for fname in sample_list:
        fpath = os.path.join(folder,fname)
        if fpath.split('.')[-1] == ftype:
            #read image
            img = read_image(fpath=fpath, ftype=ftype)
            # crop
            rows, cols = img.shape
            row_cnt = int(np.round(rows/2))
            col_cnt = int(np.round(cols/2))
            row_crop_half = int(np.round(crop_box[0]/2))
            col_crop_half = int(np.round(crop_box[1]/2))
            img_crop = img[row_cnt-row_crop_half:
                        row_cnt+row_crop_half,
                        col_cnt-col_crop_half:
                        col_cnt+col_crop_half]
            # resize
            img_rsz = transform.resize(
                img_crop/255,res_box,mode='reflect')
            if clipflag:
                img_rsz = get_sigma_clip(img_rsz,
                                         sigma=clipparam[0],
                                         iters=clipparam[1])
            # push into sample_mat
            img_vec = img_rsz.reshape((res_box[0]*res_box[1],))
            sample_mat[idx,:] = img_vec
            idx = idx + 1
        else:
            continue

    # save
    if not savepath is None:
        stype = savepath.split('.')[-1]
        if stype == 'mat':
            # save as mat
            sample_dict = {'data':sample_mat,
                           'name':sample_list}
            sio.savemat(savepath,sample_dict)
        elif stype == 'pkl':
            fp = open(savepath,'wb')
            sample_dict = {'data':sample_mat,
                           'name':sample_list}
            pickle.dump(sample_dict,fp)
            fp.close()

    return sample_mat

def load_sample(samplepath):
    """Load the sample matrix

    input
    =====
    samplepath: str
        Path to save the samples
    """
    ftype = samplepath.split('.')[-1]
    if ftype == 'pkl':
        try:
            fp = open(samplepath, 'rb')
        except:
            return None
        sample_dict = pickle.load(fp)
        sample_mat = sample_dict['data']
        sample_list = sample_dict['name']
    elif ftype == 'mat':
        try:
            sample_dict = sio.loadmat(samplepath)
        except:
            return None
        sample_mat = sample_dict['data']
        sample_list = sample_dict['name']

    return sample_mat, sample_list

def get_sigma_clip(img,sigma=3,iters=100):
    """
    Do sigma clipping on the raw images to improve constrast of
    target regions.

    Reference
    =========
    [1] sigma clip
        http://docs.astropy.org/en/stable/api/astropy.stats.sigma_clip.html
    """
    img_clip = sigma_clip(img, sigma=sigma, iters=iters)
    img_mask = img_clip.mask.astype(float)
    img_new = img * img_mask

    return img_new

def get_augmentation(img, crop_box=(150,150), rez_box=(50,50),
                     num_aug = 1, clipflag=False,clipparam=None):
    """
    Do image augmentation

    References
    ==========
    [1] http://blog.sina.com.cn/s/blog_5562b04401015bys.html
    [2] http://blog.csdn.net/guduruyu/article/details/70842142

    steps
    =====
    flip -> rotate -> crop -> resize

    inputs
    ======
    img: np.ndarray or str
        image or the image path
    crop_box: tuple
        Size of the crop box
    rez_box: tuple
        Size of the resized box

    output
    ======
    img_aug: augmented image
    """
    from PIL import Image
    from astropy.io import fits
    # load image
    if isinstance(img, str):
        if img.split(".")[-1] == "fits":
            h = fits.open(img)
            h = h[0].data
            h = np.nan_to_num(h)
            h = (h-h.min())/(h.max()-h.min())
            img_raw = Image.fromarray(h)
        else:
            img_raw = Image.open(img)
            img_raw = img_raw.convert('L')
    else:
        img_raw = Image.fromarray(img)
        # img_raw = img_raw.convert('L')

    # sigma clipping
    if clipflag == True:
        img_raw = get_sigma_clip(np.array(img_raw),
                                 sigma=clipparam[0],
                                 iters=clipparam[1])
        img_raw = Image.fromarray(img_raw)
    # rbg2grey
    img_r = np.zeros((num_aug, rez_box[0], rez_box[1]))
    for i in range(num_aug):
        # flip
        idx = np.random.permutation(2)[0]
        if idx == 0:
            img_aug = img_raw.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img_aug = img_raw.transpose(Image.FLIP_TOP_BOTTOM)

        # rotate
        angle = np.random.uniform() * 360
        img_aug = img_aug.rotate(angle, expand=True)

        # crop
        rows = img_aug.width
        cols = img_aug.height
        row_cnt = int(np.round(rows/2))
        col_cnt = int(np.round(cols/2))
        row_crop_half = int(np.round(crop_box[0]/2))
        col_crop_half = int(np.round(crop_box[1]/2))
        crop_tuple = (row_cnt-row_crop_half,
                      col_cnt-col_crop_half,
                      row_cnt+row_crop_half,
                      col_cnt+col_crop_half)
        img_aug = img_aug.crop(box=crop_tuple)

        # resize
        img_aug = img_aug.resize(rez_box)

        # Image to matrix
        img_r[i,:,:] = np.array(img_aug)

    return img_r

def get_augmentation_single(img, crop_box=(150,150), rez_box=(50,50),
                     num_aug = 1, clipflag=False,clipparam=None):
    """
    Do image augmentation, only clipping

    References
    ==========
    [1] http://blog.sina.com.cn/s/blog_5562b04401015bys.html
    [2] http://blog.csdn.net/guduruyu/article/details/70842142

    steps
    =====
    flip -> rotate -> crop -> resize

    inputs
    ======
    img: np.ndarray or str
        image or the image path
    crop_box: tuple
        Size of the crop box
    rez_box: tuple
        Size of the resized box

    output
    ======
    img_aug: augmented image
    """
    from PIL import Image
    from astropy.io import fits
    # load image
    if isinstance(img, str):
        if img.split(".")[-1] == "fits":
            h = fits.open(img)
            h = h[0].data
            h = np.nan_to_num(h)
            h = (h-h.min())/(h.max()-h.min())
            img_raw = Image.fromarray(h)
        else:
            img_raw = Image.open(img)
            img_raw = img_raw.convert('L')
    else:
        img_raw = Image.fromarray(img)
        # img_raw = img_raw.convert('L')

    # sigma clipping
    if clipflag == True:
        img_raw = get_sigma_clip(np.array(img_raw),
                                 sigma=clipparam[0],
                                 iters=clipparam[1])
        img_raw = Image.fromarray(img_raw)
    # rbg2grey
    img_r = np.zeros((num_aug, rez_box[0], rez_box[1]))
    for i in range(num_aug):
        # crop
        rows = img_raw.width
        cols = img_raw.height
        row_cnt = int(np.round(rows/2))
        col_cnt = int(np.round(cols/2))
        row_crop_half = int(np.round(crop_box[0]/2))
        col_crop_half = int(np.round(crop_box[1]/2))
        crop_tuple = (row_cnt-row_crop_half,
                      col_cnt-col_crop_half,
                      row_cnt+row_crop_half,
                      col_cnt+col_crop_half)
        img_aug = img_raw.crop(box=crop_tuple)

        # resize
        img_aug = img_aug.resize(rez_box)

        # Image to matrix
        img_r[i,:,:] = np.array(img_aug)

    return img_r

def get_predict(sess,names,img,input_shape=[140,140,1]):
    """
    Predict the output of the input image

    input
    =====
    sess: tf.Session()
        The saved session
    names: dict
        The dict saved names of the variables.
    img: np.ndarray
        The image matrix, (r,c)

    output
    ======
    img_pred: np.ndarray
        The predicted image matrix
    """
    if img.dtype != 'float32':
        img = img.astype('float32')

    # params
    depth = input_shape[2]
    rows = input_shape[0]
    cols = input_shape[1]
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
    # sess.run(tf.global_variables_initializer())
    img_pred = sess.run(names['l_de'], feed_dict={names['l_in']: img_te, names['droprate']: 0.})

    return img_pred

def get_encode(sess, names, img, input_shape=[28,28,1]):
    """
    Generate the codes of the input image

    input
    =====
    sess: tf.Session()
        The saved session
    names: dict
        The dict saved names of the variables.
    img: np.ndarray
        The image matrix, (r,c)

    output
    ======
    code: np.ndarray
        The codes
    """
    if img.dtype != 'float32':
        img = img.astype('float32')

    # params
    depth = input_shape[2]
    rows = input_shape[0]
    cols = input_shape[1]
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
    # sess.run(tf.global_variables_initializer())
    code = sess.run(names['l_en'], feed_dict={names['l_in']: img_te, names['droprate']: 0.})

    return code


def get_decode(sess, names, code, input_shape = [28,28,1], code_len=32):
    """Decode to output the recovered image

    input
    =====
    sess: tf.Session()
        The saved session
    names: dict
        The dict saved names of the variables.
    code: np.ndarray
        The code to be decoded.

    output
    ======
    img_de: np.ndarray
        The recovered or predicted image matrix
    """
    # Compare code length
    if code.shape[1] != code_len:
        print("The length of provided codes should be equal to the network's")
        return None
    else:
        # decoding
        l_in_shape = [code.shape[0]]
        l_in_shape.extend(input_shape)
        p_in = np.zeros(l_in_shape) # pseudo input
        img_de = sess.run(names['l_de'],
                          feed_dict={names['l_in']: p_in,
                                     names['l_en']: code,
                                     names['droprate']: 0.})

    return img_de

def load_net(namepath):
    """
    Load the cae network

    reference
    =========
    [1] https://www.cnblogs.com/azheng333/archive/2017/06/09/6972619.html

    input
    =====
    namepath: str
        Path to save the trained network

    output
    ======
    sess: tf.Session()
        The restored session
    names: dict
        The dict saved variables names
    """
    try:
        fp = open(namepath,'rb')
    except:
        return None

    names = pickle.load(fp)

    # load the net
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, names['netpath'])

    return sess, names

def sort_sample(inpath, savepath=None):
    """
    Sort the samples' names which are namely with integers, into descending
    sort, as well as the corresponding image data.

    Inputs
    ======
    inpath: str
        Path of the unsorted samples
    outpath: str
        Path of the sorted ones, if set as None, the inpath will be used.
    """
    with open(inpath, 'rb') as fp:
        datadict = pickle.load(fp)

    data = datadict['data']
    name = datadict['name']

    # rename the files with equai-length name
    numsamples = data.shape[0]
    # maximun number of characters in a sample name
    maxch = len(str(numsamples))
    # rename
    for i in range(numsamples):
        s = name[i]
        n = s.split('.')
        # fill
        n[0] = '0'*(maxch - len(n[0])) + n[0]
        # rename
        name[i] = '.'.join(n)

    # sort
    idx_sort = np.argsort(name)
    name_array = np.array(name)

    name_sort = name_array[idx_sort]
    data_sort = data[idx_sort]

    # save
    datadict['name'] = name_sort
    datadict['data'] = data_sort

    if savepath is None:
        savepath = inpath

    with open(savepath, 'wb') as f:
        pickle.dump(datadict, f)

def down_dimension(code, method='tSNE', params=None):
    """
    Do dimension decreasing of the codes, so as to evaluate samples'
    distributions.

    Inputs
    ======
    code: np.ndarray
        The estimated codes by the cae net on the samples.
    method: str
        The method of dimension decreasing, could be PCA, tSNE or Kmeans,
        default as PCA.
    params: dict
        Corresponding parameters to the method.

    Output
    ======
    code_dim: np.ndarray
    The dimension decreased matrix.
    """
    if method == 'PCA':
        from sklearn.decomposition import PCA
        code_dim = PCA().fit_transform(code)
    elif method == 'tSNE':
        from sklearn.manifold import TSNE
        tsne = TSNE()
        for key in params.keys():
            try:
                setattr(tsne, key, params['key'])
            except:
                continue
        code_dim = tsne.fit_transform(code)
    elif method == 'Kmeans':
        from sklearn.cluster import KMeans
        code_dim = KMeans()
        for key in params.keys():
            try:
                setattr(code_dim, key, params['key'])
            except:
                continue
        code_dim.fit(code)
    else:
        print("The method %s is not supported at present." % method)

    return code_dim

def sub2multi(data,label,mask):
    """Subtypes to multiple categories according to provided mask."""
    label_bin = label
    for i,m in enumerate(mask):
        label_bin[np.where(label == i)] = m
    # remove samples of label larger than 1, i.e., discarded samples
    idx = np.where(label_bin <= 2)[0]
    return data[idx,:],label[idx]

def vec2onehot(label,numclass):
    label_onehot = np.zeros((len(label),numclass))
    for i,l in enumerate(label):
        label_onehot[i, int(l)] = 1

    return label_onehot
