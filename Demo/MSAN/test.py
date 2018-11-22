from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import numpy as np
import configuration

from datasets import load_dataset
from model import LTS
from recall import i2t
from vocab import build_dictionary

from collections import OrderedDict, defaultdict

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_dir", "", "Directory containing model checkpoints.")
tf.flags.DEFINE_string("dataset_dir", "", "Directory containing dataset.")
tf.flags.DEFINE_string("input_dataset_name", "coco", "name of the dataset")

def test_model():
    model_config = configuration.ModelConfig()
    model_config.data = FLAGS.input_dataset_name

    print ('Loading %s dataset ...' % FLAGS.input_dataset_name)
    (train_caps, train_ims_local, train_ims_global), (test_caps, test_ims_local, test_ims_global) = load_dataset(
            path=FLAGS.dataset_dir, name=model_config.data, load_train=True)

    total_caps = train_caps + test_caps
    total_ims_local = np.concatenate((train_ims_local, test_ims_local), axis=0)
    total_ims_global = np.concatenate((train_ims_global[:, :4096], test_ims_global[:, :4096]), axis=0)
    total_ims_NIC = np.concatenate((train_ims_global[:, 4096:], test_ims_global[:, 4096:]), axis=0)

    print ('creating dictionary')
    worddict = build_dictionary(total_caps)[0]
    n_words = len(worddict)
    model_config.n_words = n_words
    model_config.worddict = worddict
    print ('dictionary size: ' + str(n_words))

    print ('Building the model ...')
    model = LTS(model_config)
    model.build()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if not model_path:
            print("Skipping testing. No checkpoint found in: %s" % FLAGS.checkpoint_dir)
            return

        print("Loading model from checkpoint: %s" % model_path)
        saver.restore(sess, model_path)
        print("Successfully loaded checkpoint: %s" % model_path)

        images = getTestImageFeature(sess, model, model_config, total_ims_local, total_ims_global, total_ims_NIC)
        ls = getTestTextFeature(sess, model, model_config, total_caps)
        np.save("image_embeddings.npy", images)
        np.save("text_embeddings.npy", ls)

def main():
    assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
    assert FLAGS.dataset_dir, "--dataset_dir is required"

    test_model()


def getTestImageFeature(sess, model, model_config, local_feature, global_feature, NIC_feature):
    features = np.zeros((local_feature.shape[0], model_config.dim*3), dtype=np.float32)

    numbatches = features.shape[0] // model_config.batch_size + 1
    len_indices_pos = 0
    len_curr_counts = features.shape[0]
    for minibatch in range(numbatches):
        curr_batch_size = np.minimum(model_config.batch_size, len_curr_counts)
        curr_indices = range(len_indices_pos, len_indices_pos+curr_batch_size)
        vgg_local_feature = local_feature[curr_indices]

        images = sess.run(model.vfeats_all,
                            feed_dict={model.VGG_local_pred_data: local_feature[curr_indices],
                                       model.VGG_global_pred_data: global_feature[curr_indices],
                                       model.NIC_pred_data: NIC_feature[curr_indices],
                                       model.keep_prob: 1.0,
                                       model.phase: 0})

        bv = images[0].shape[0]
        vfeat_size = images[0].shape[1]
        v_step = len(images)
        vfeats = np.zeros([bv, vfeat_size*v_step])
        for i, vfeat_single in enumerate(images):
            features[curr_indices, i*vfeat_size:i*vfeat_size+vfeat_size] = vfeat_single

        len_indices_pos += curr_batch_size
        len_curr_counts -= curr_batch_size

    return features


def getTestTextFeature(sess, model, model_config, test_caps):
    features = np.zeros((len(test_caps), model_config.dim*3), dtype=np.float32)

    # length dictionary
    ds = defaultdict(list)

    captions = []
    for s in test_caps:
        s = s.lower()
        captions.append(s.split())

    for i,s in enumerate(captions):
        ds[len(s)].append(i)

    #quick check if a word is in the dictionary
    d = defaultdict(lambda : 0)
    for w in model_config.worddict.keys():
        d[w] = 1

    # Get features
    for k in ds.keys():
        numbatches = len(ds[k]) // model_config.batch_size + 1
        for minibatch in range(numbatches):
            caps = ds[k][minibatch::numbatches]
            caption = [captions[c] for c in caps]

            seqs = []
            for i, cc in enumerate(caption):
                seqs.append([model_config.worddict[w] if w in model_config.worddict.keys() else 1 for w in cc])

            x = np.zeros((k+1, len(caption))).astype('int64')
            x_mask = np.zeros((k+1, len(caption))).astype('float32')
            for idx, s in enumerate(seqs):
                x[:k,idx] = s
                x_mask[:k+1,idx] = 1.

            ff = sess.run(model.sfeats_all,
                            feed_dict={model.ls_pred_data: x.T,
                                       model.input_mask: x_mask.T,
                                       model.keep_prob: 1.0,
                                       model.phase: 0})

            bs = ff[0].shape[0]
            sfeat_size = ff[0].shape[1]
            s_step = len(ff)
            sfeats = np.zeros([bs, sfeat_size*s_step])
            for i,sfeat_single in enumerate(ff):
                sfeats[:,i*sfeat_size:i*sfeat_size+sfeat_size] = sfeat_single

            for ind, c in enumerate(caps):
                features[c] = sfeats[ind]

    return features

if __name__ == '__main__':
    main()
