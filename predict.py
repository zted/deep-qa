from datetime import datetime
from sklearn import metrics
from theano import tensor as T
import cPickle
import numpy
import os
import sys
import theano
import time
from collections import defaultdict
import subprocess
import pandas as pd
from tqdm import tqdm

import nn_layers
import sgd_trainer

import warnings
warnings.filterwarnings("ignore")  # TODO remove

### THEANO DEBUG FLAGS
# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity = 'high'


def main():
    # ZEROUT_DUMMY_WORD = False
    ZEROUT_DUMMY_WORD = True

    ## Load data
    # mode = 'TRAIN-ALL'
    mode = 'train'
    print "Running training in the {} setting".format(mode)

    data_dir = '/home/ted/research/deep-qa/TRAIN'

    q_train = numpy.load(os.path.join(data_dir, 'train.questions.npy'))
    a_train = numpy.load(os.path.join(data_dir, 'train.answers.npy'))
    q_overlap_train = numpy.load(os.path.join(data_dir, 'train.q_overlap_indices.npy'))
    a_overlap_train = numpy.load(os.path.join(data_dir, 'train.a_overlap_indices.npy'))
    y_train = numpy.load(os.path.join(data_dir, 'train.labels.npy'))

    q_test = numpy.load(os.path.join(data_dir, 'test.questions.npy'))
    a_test = numpy.load(os.path.join(data_dir, 'test.answers.npy'))
    q_overlap_test = numpy.load(os.path.join(data_dir, 'test.q_overlap_indices.npy'))
    a_overlap_test = numpy.load(os.path.join(data_dir, 'test.a_overlap_indices.npy'))
    y_test = numpy.load(os.path.join(data_dir, 'test.labels.npy'))
    qids_test = numpy.load(os.path.join(data_dir, 'test.qids.npy'))
    pids_test = numpy.load(os.path.join(data_dir, 'test.pids.npy'))

    print 'y_train', numpy.unique(y_train, return_counts=True)
    print 'y_test', numpy.unique(y_test, return_counts=True)

    print 'q_train', q_train.shape
    print 'q_test', q_test.shape

    print 'a_train', a_train.shape
    print 'a_test', a_test.shape

    numpy_rng = numpy.random.RandomState(123)
    q_max_sent_size = q_train.shape[1]
    a_max_sent_size = a_train.shape[1]

    ndim = 5
    print "Generating random vocabulary for word overlap indicator features with dim:", ndim
    dummy_word_id = numpy.max(a_overlap_train)
    print "Gaussian"
    vocab_emb_overlap = numpy_rng.randn(dummy_word_id+1, ndim) * 0.25
    vocab_emb_overlap[-1] = 0

    # Load word2vec embeddings
    fname = os.path.join(data_dir, 'emb_aquaint+wiki.txt.gz.ndim=50.bin.npy')

    print "Loading word embeddings from", fname
    vocab_emb = numpy.load(fname)
    ndim = vocab_emb.shape[1]
    dummpy_word_idx = numpy.max(a_train)
    print "Word embedding matrix size:", vocab_emb.shape

    x = T.dmatrix('x')
    x_q = T.lmatrix('q')
    x_q_overlap = T.lmatrix('q_overlap')
    x_a = T.lmatrix('a')
    x_a_overlap = T.lmatrix('a_overlap')
    y = T.ivector('y')

    #######
    n_outs = 2

    n_epochs = 2
    batch_size = 50
    learning_rate = 0.1
    max_norm = 0

    print 'batch_size', batch_size
    print 'n_epochs', n_epochs
    print 'learning_rate', learning_rate
    print 'max_norm', max_norm

    ## 1st conv layer.
    ndim = vocab_emb.shape[1] + vocab_emb_overlap.shape[1]

    ### Nonlinearity type
    # activation = nn_layers.relu_f
    activation = T.tanh

    dropout_rate = 0.5
    nkernels = 100
    q_k_max = 1
    a_k_max = 1

    # filter_widths = [3,4,5]
    q_filter_widths = [5]
    a_filter_widths = [5]

    ###### QUESTION ######
    lookup_table_words = nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(q_filter_widths)-1)
    lookup_table_overlap = nn_layers.LookupTableFast(W=vocab_emb_overlap, pad=max(q_filter_widths)-1)

    lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words, lookup_table_overlap])

    num_input_channels = 1
    input_shape = (batch_size, num_input_channels, q_max_sent_size + 2*(max(q_filter_widths)-1), ndim)

    conv_layers = []
    for filter_width in q_filter_widths:
        filter_shape = (nkernels, num_input_channels, filter_width, ndim)
        conv = nn_layers.Conv2dLayer(rng=numpy_rng, filter_shape=filter_shape, input_shape=input_shape)
        non_linearity = nn_layers.NonLinearityLayer(b_size=filter_shape[0], activation=activation)
        pooling = nn_layers.KMaxPoolLayer(k_max=q_k_max)
        conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[conv, non_linearity, pooling])
        conv_layers.append(conv2dNonLinearMaxPool)

    join_layer = nn_layers.ParallelLayer(layers=conv_layers)
    flatten_layer = nn_layers.FlattenLayer()

    nnet_q = nn_layers.FeedForwardNet(layers=[
        lookup_table,
        join_layer,
        flatten_layer,
    ])
    nnet_q.set_input((x_q, x_q_overlap))
    ######


    ###### ANSWER ######
    lookup_table_words = nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(q_filter_widths)-1)
    lookup_table_overlap = nn_layers.LookupTableFast(W=vocab_emb_overlap, pad=max(q_filter_widths)-1)

    lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words, lookup_table_overlap])

    input_shape = (batch_size, num_input_channels, a_max_sent_size + 2*(max(a_filter_widths)-1), ndim)
    conv_layers = []
    for filter_width in a_filter_widths:
        filter_shape = (nkernels, num_input_channels, filter_width, ndim)
        conv = nn_layers.Conv2dLayer(rng=numpy_rng, filter_shape=filter_shape, input_shape=input_shape)
        non_linearity = nn_layers.NonLinearityLayer(b_size=filter_shape[0], activation=activation)
        pooling = nn_layers.KMaxPoolLayer(k_max=a_k_max)
        conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[conv, non_linearity, pooling])
        conv_layers.append(conv2dNonLinearMaxPool)

    join_layer = nn_layers.ParallelLayer(layers=conv_layers)
    flatten_layer = nn_layers.FlattenLayer()

    nnet_a = nn_layers.FeedForwardNet(layers=[
        lookup_table,
        join_layer,
        flatten_layer,
    ])
    nnet_a.set_input((x_a, x_a_overlap))

    q_logistic_n_in = nkernels * len(q_filter_widths) * q_k_max
    a_logistic_n_in = nkernels * len(a_filter_widths) * a_k_max

    pairwise_layer = nn_layers.PairwiseNoFeatsLayer(q_in=q_logistic_n_in,
                                                    # pairwise_layer = nn_layers.PairwiseWithFeatsLayer(q_in=q_logistic_n_in,
                                                    # pairwise_layer = nn_layers.PairwiseOnlySimWithFeatsLayer(q_in=q_logistic_n_in,
                                                    a_in=a_logistic_n_in)
    pairwise_layer.set_input((nnet_q.output, nnet_a.output))

    n_in = q_logistic_n_in + a_logistic_n_in + 1
    hidden_layer = nn_layers.LinearLayer(numpy_rng, n_in=n_in, n_out=n_in, activation=activation)
    hidden_layer.set_input(pairwise_layer.output)

    classifier = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)
    classifier.set_input(hidden_layer.output)


    train_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, pairwise_layer, hidden_layer, classifier],
                                          # train_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, x_hidden_layer, classifier],
                                          name="Training nnet")
    test_nnet = train_nnet
    #######

    print train_nnet

    params = train_nnet.params

    ts = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    nnet_outdir = 'exp.out/ndim={};batch={};max_norm={};learning_rate={};{}'.format(ndim, batch_size, max_norm, learning_rate, ts)
    if not os.path.exists(nnet_outdir):
        os.makedirs(nnet_outdir)
    nnet_fname = os.path.join(nnet_outdir, 'nnet.dat')
    print "Saving to", nnet_fname
    cPickle.dump([train_nnet, test_nnet], open(nnet_fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)


    total_params = sum([numpy.prod(param.shape.eval()) for param in params])
    print 'Total params number:', total_params

    cost = train_nnet.layers[-1].training_cost(y)
    predictions = test_nnet.layers[-1].y_pred
    predictions_prob = test_nnet.layers[-1].p_y_given_x[:,-1]

    # batch_x = T.dmatrix('batch_x')
    batch_x_q = T.lmatrix('batch_x_q')
    batch_x_a = T.lmatrix('batch_x_a')
    batch_x_q_overlap = T.lmatrix('batch_x_q_overlap')
    batch_x_a_overlap = T.lmatrix('batch_x_a_overlap')
    batch_y = T.ivector('batch_y')

    # updates = sgd_trainer.get_adagrad_updates(cost, params, learning_rate=learning_rate, max_norm=max_norm, _eps=1e-6)
    updates = sgd_trainer.get_adadelta_updates(cost, params, rho=0.95, eps=1e-6, max_norm=max_norm, word_vec_name='W_emb')

    inputs_pred = [batch_x_q,
                   batch_x_a,
                   batch_x_q_overlap,
                   batch_x_a_overlap,
                   # batch_x,
                   ]

    givens_pred = {x_q: batch_x_q,
                   x_a: batch_x_a,
                   x_q_overlap: batch_x_q_overlap,
                   x_a_overlap: batch_x_a_overlap,
                   # x: batch_x
                   }

    pred_fn = theano.function(inputs=inputs_pred,
                              outputs=predictions,
                              givens=givens_pred)

    pred_prob_fn = theano.function(inputs=inputs_pred,
                                   outputs=predictions_prob,
                                   givens=givens_pred)

    def predict_prob_batch(batch_iterator):
        preds = numpy.hstack([pred_prob_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap) for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, _ in batch_iterator])
        return preds[:batch_iterator.n_samples]

    test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, y_test], batch_size=batch_size, randomize=False)

    labels = sorted(numpy.unique(y_test))
    print 'labels', labels

    print "Zero out dummy word:", ZEROUT_DUMMY_WORD
    if ZEROUT_DUMMY_WORD:
        W_emb_list = [w for w in params if w.name == 'W_emb']
        zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])

    best_dev_acc = -numpy.inf
    epoch = 0
    timer_train = time.time()

    model_file = open('./exp.out/best_dev_params_combined', 'rb')
    best_params = cPickle.load(model_file)
    model_file.close()

    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))
    for i, param in enumerate(best_params):
        params[i].set_value(param, borrow=True)

    y_pred_test = predict_prob_batch(test_set_iterator)
    # test_acc = map_score(qids_test, y_test, y_pred_test) * 100
    fname = os.path.join(nnet_outdir, 'best_dev_params.epoch={:02d};batch={:05d};dev_acc={:.2f}.dat'.format(epoch, i, best_dev_acc))
    numpy.savetxt(os.path.join(nnet_outdir, 'test.epoch={:02d};batch={:05d};dev_acc={:.2f}.predictions.npy'.format(epoch, i, best_dev_acc)), y_pred_test)
    cPickle.dump(best_params, open(fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)


    print "Running trec_eval script..."
    N = len(y_pred_test)

    df_submission = pd.DataFrame(index=numpy.arange(N), columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
    df_submission['qid'] = qids_test
    df_submission['iter'] = 0
    df_submission['docno'] = pids_test
    df_submission['rank'] = 0
    df_submission['sim'] = y_pred_test
    df_submission['run_id'] = 'nnet'
    df_submission.to_csv(os.path.join(nnet_outdir, 'submission.txt'), header=False, index=False, sep=' ')

    # df_gold = pd.DataFrame(index=numpy.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
    # df_gold['qid'] = qids_test
    # df_gold['iter'] = 0
    # df_gold['docno'] = numpy.arange(N)
    # df_gold['rel'] = y_test
    # df_gold.to_csv(os.path.join(nnet_outdir, 'gold.txt'), header=False, index=False, sep=' ')

    # subprocess.call("/bin/sh run_eval.sh '{}'".format(nnet_outdir), shell=True)

if __name__ == '__main__':
    main()
