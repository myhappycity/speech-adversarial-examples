from tensorflow.core.framework.graph_pb2 import *
import numpy as np
import tensorflow as tf
import sys
from util.audio import audiofile_to_input_vector
from util.text import ctc_label_dense_to_sparse

tf.load_op_library = lambda x: x
import DeepSpeech as DeepSpeech

graph_def = GraphDef()
loaded = graph_def.ParseFromString(open("models/output_graph.pb","rb").read())

with tf.Graph().as_default() as graph:
    new_input = tf.placeholder(tf.float32, [None, None, None],
                               name="new_input")
    # Load the saved .pb into the current graph to let us grab
    # access to the weights.
    logits, = tf.import_graph_def(
        graph_def,
        input_map={"input_node:0": new_input},
        return_elements=['logits:0'],
        name="newname",
        op_dict=None,
        producer_op_list=None
    )

   
    with tf.Session(graph=graph) as sess:
        # Sample sentetnce, to make sure we've done it right
        mfcc = audiofile_to_input_vector("sample_input.wav", 26, 9)

       
        tf.app.flags.FLAGS.alphabet_config_path = "DeepSpeech/data/alphabet.txt"
        DeepSpeech.initialize_globals()
        logits2 = DeepSpeech.BiRNN(new_input, [len(mfcc)], [0]*10)

    
        for var in tf.global_variables():
            sess.run(var.assign(sess.run('newname/'+var.name)))

       
        res = (sess.run(logits, {new_input: [mfcc],
                                     'newname/input_lengths:0': [len(mfcc)]}).flatten())
        res2 = (sess.run(logits2, {new_input: [mfcc]})).flatten()
        print('This value should be small',np.sum(np.abs(res-res2)))

       
        saver = tf.train.Saver()
        saver.save(sess, "models/session_dump")
