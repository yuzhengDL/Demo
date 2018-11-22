# -*- coding: utf-8 -*-
import sys
sys.path.append("..")

import cPickle as pkl
import os.path
import random
import re
try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

import numpy as np
import gensim
import jieba
from skimage import io

from sklearn.cluster import KMeans

import faiss
import tensorflow as tf

import MSAN.configuration as configuration
from MSAN.model import LTS
from vgg19 import vgg_exactor

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')

class MungedPageHandler(tornado.web.RequestHandler):

    def post(self):
        input_text = self.get_argument('words')

        # input image URL
        if input_text in image_names:
            ims_embedding = name2feat[input_text]
            idx = name2index[input_text]
            gt = range(idx*5, idx*5+5)

            # faiss search
            _, I = pq_ls.search(ims_embedding.reshape([1, -1]), 50)
            I = I.flatten()

            self.render('Image2Text.html', ims_url=input_text, w=I, index=caps, words=input_text, groudtruth=gt)
        elif input_text:
            # input sentence/word
            sent = input_text.lower().split()
            sent = [model_config.worddict[w] if w in model_config.worddict.keys() else 1 for w in sent]

            k = len(sent)
            x = np.zeros((1, k+1)).astype('int64')
            x_mask = np.zeros((1, k+1)).astype('float32')

            x[:, :k] = sent
            x_mask[:, :k+1] = 1.

            text_list = sess.run(model.sfeats_all,
                                    feed_dict={model.ls_pred_data: x,
                                               model.input_mask: x_mask,
                                               model.keep_prob: 1.0,
                                               model.phase: 0})
            sfeat_size = text_list[0].shape[1]
            s_step = len(text_list)
            sfeats = np.zeros([1, sfeat_size*s_step]).astype('float32')
            for i, sfeat_single in enumerate(text_list):
                sfeats[:, i*sfeat_size:i*sfeat_size+sfeat_size] = sfeat_single

            # Top 20 results
            _, I = pq_ims.search(sfeats, 50)
            I = I.flatten()

            self.render('Text2Img.html', words=input_text, index_results=I, total_names=image_names)

        else:
            self.render('index.html')

model_config = configuration.ModelConfig()

# load text embeddings from database
caps = []
with open('/media/amax/yz/dataset/mscoco/coco_VGG/coco_train_caps.txt') as f:
    for line in f.readlines():
        caps.append(line.strip())

text_embeddings = np.load("../MSAN/text_embeddings.npy")[:413915]
with open('../MSAN/coco.dictionary.pkl') as f:
    worddict = pkl.load(f)
model_config.n_words = len(worddict)
model_config.worddict = worddict

# load image embeddings from database
image_embeddings = np.load("../MSAN/image_embeddings.npy")[:82783]
image_names = []
name2feat = {}
name2index = {}
with open('/media/amax/yz/dataset/mscoco/coco_VGG/coco_train.txt') as f:
    for i, line in enumerate(f.readlines()):
        image_names.append(line.strip())
        name2feat[line.strip()] = image_embeddings[i]
        name2index[line.strip()] = i

# initialize MSAN
model = LTS(model_config)
model.build()
saver = tf.train.Saver()
model_path = tf.train.latest_checkpoint('/media/amax/yz/desktop/VGG_attention/coco/checkpoint_files/')
sess = tf.Session()
print("Loading model from checkpoint: %s" % model_path)
saver.restore(sess, model_path)
print("Successfully loaded checkpoint: %s" % model_path)

# initialize VGG19
with tf.Graph().as_default():
    model_vgg = vgg_exactor()
    sess_vgg = tf.Session()
    model_vgg.init_fn(sess_vgg)

# creat faiss index
k = 10
m = 32
n_bits = 8
pq_ims = faiss.IndexPQ(model_config.dim*3, m, n_bits)
pq_ims.train(image_embeddings)
pq_ims.add(image_embeddings)

pq_ls = faiss.IndexPQ(model_config.dim*3, m, n_bits)
pq_ls.train(text_embeddings)
pq_ls.add(text_embeddings)

print "Server initialization complete."

tornado.options.parse_command_line()
app = tornado.web.Application(
    handlers=[(r'/', IndexHandler), (r'/rank', MungedPageHandler)],
    template_path=os.path.join(os.path.dirname(__file__), "templates"),
    static_path=os.path.join(os.path.dirname(__file__), "static"),
    debug=True
)
http_server = tornado.httpserver.HTTPServer(app)
http_server.listen(options.port)
tornado.ioloop.IOLoop.instance().start()

