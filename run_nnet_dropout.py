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

    data_dir = 'TRAIN'

    q_train = numpy.load(os.path.join(data_dir, 'train.questions.npy'))
    a_train = numpy.load(os.path.join(data_dir, 'train.answers.npy'))
    q_overlap_train = numpy.load(os.path.join(data_dir, 'train.q_overlap_indices.npy'))
    a_overlap_train = numpy.load(os.path.join(data_dir, 'train.a_overlap_indices.npy'))
    y_train = numpy.load(os.path.join(data_dir, 'train.labels.npy'))

    q_dev = numpy.load(os.path.join(data_dir, 'dev.questions.npy'))
    a_dev = numpy.load(os.path.join(data_dir, 'dev.answers.npy'))
    q_overlap_dev = numpy.load(os.path.join(data_dir, 'dev.q_overlap_indices.npy'))
    a_overlap_dev = numpy.load(os.path.join(data_dir, 'dev.a_overlap_indices.npy'))
    y_dev = numpy.load(os.path.join(data_dir, 'dev.labels.npy'))

    q_test = numpy.load(os.path.join(data_dir, 'test.questions.npy'))
    a_test = numpy.load(os.path.join(data_dir, 'test.answers.npy'))
    q_overlap_test = numpy.load(os.path.join(data_dir, 'test.q_overlap_indices.npy'))
    a_overlap_test = numpy.load(os.path.join(data_dir, 'test.a_overlap_indices.npy'))
    y_test = numpy.load(os.path.join(data_dir, 'test.labels.npy'))
    qids_dev = numpy.load(os.path.join(data_dir, 'dev.qids.npy'))

    print 'y_train', numpy.unique(y_train, return_counts=True)
    print 'y_dev', numpy.unique(y_dev, return_counts=True)
    print 'y_test', numpy.unique(y_test, return_counts=True)

    print 'q_train', q_train.shape
    print 'q_dev', q_dev.shape
    print 'q_test', q_test.shape

    print 'a_train', a_train.shape
    print 'a_dev', a_dev.shape
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

    n_epochs = 25
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

    # num_input_channels = len(lookup_table.layers)
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
    #######
    # print 'nnet_q.output', nnet_q.output.ndim

    q_logistic_n_in = nkernels * len(q_filter_widths) * q_k_max
    a_logistic_n_in = nkernels * len(a_filter_widths) * a_k_max

    dropout_q = nn_layers.FastDropoutLayer(rng=numpy_rng)
    dropout_a = nn_layers.FastDropoutLayer(rng=numpy_rng)
    dropout_q.set_input(nnet_q.output)
    dropout_a.set_input(nnet_a.output)

    pairwise_layer = nn_layers.PairwiseNoFeatsLayer(q_in=q_logistic_n_in,
                                                    a_in=a_logistic_n_in)
    pairwise_layer.set_input((dropout_q.output, dropout_a.output))

    n_in = q_logistic_n_in + a_logistic_n_in + 1

    hidden_layer = nn_layers.LinearLayer(numpy_rng, n_in=n_in, n_out=n_in, activation=activation)
    hidden_layer.set_input(pairwise_layer.output)

    dropout_layer = nn_layers.FastDropoutLayer(rng=numpy_rng)
    dropout_layer.set_input(hidden_layer.output)

    classifier = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)
    classifier.set_input(dropout_layer.output)

    train_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, dropout_q, dropout_a,
                                                  pairwise_layer, hidden_layer, dropout_layer, classifier],
                                          name="Training nnet")

    pairwise_layer_test = nn_layers.PairwiseNoFeatsLayer(q_in=q_logistic_n_in,
                                                    a_in=a_logistic_n_in)
    pairwise_layer_test.set_input((nnet_q.output, nnet_a.output))
    pairwise_layer_test.weights = pairwise_layer.weights
    pairwise_layer_test.biases = pairwise_layer.biases

    hidden_layer_test = nn_layers.LinearLayer(numpy_rng, n_in=n_in, n_out=n_in, activation=activation)
    hidden_layer_test.set_input(pairwise_layer_test.output)
    hidden_layer_test.weights = hidden_layer.weights
    hidden_layer_test.biases = hidden_layer.biases

    classifier_test = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)
    classifier_test.set_input(hidden_layer_test.output)
    classifier_test.weights = classifier.weights
    classifier_test.biases = classifier.biases

    # test_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, pairwise_layer_test, hidden_layer_test, classifier_test],
    #                                       name="Predicting nnet")

    test_nnet = train_nnet

    print train_nnet

    params = train_nnet.params
    params_test = test_nnet.params

    ts = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    if len(sys.argv) > 1:
        nnet_outdir = 'exp.out/{}'.format(sys.argv[1])
    else:
        nnet_outdir = 'exp.out/ndim={};batch={};max_norm={};learning_rate={};{}'.format(ndim, batch_size, max_norm, learning_rate, ts)
    if not os.path.exists(nnet_outdir):
        os.makedirs(nnet_outdir)

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

    inputs_train = [batch_x_q,
                    batch_x_a,
                    batch_x_q_overlap,
                    batch_x_a_overlap,
                    # batch_x,
                    batch_y,
                    ]

    givens_train = {x_q: batch_x_q,
                    x_a: batch_x_a,
                    x_q_overlap: batch_x_q_overlap,
                    x_a_overlap: batch_x_a_overlap,
                    # x: batch_x,
                    y: batch_y}

    train_fn = theano.function(inputs=inputs_train,
                               outputs=cost,
                               updates=updates,
                               givens=givens_train)

    pred_fn = theano.function(inputs=inputs_pred,
                              outputs=predictions,
                              givens=givens_pred)

    pred_prob_fn = theano.function(inputs=inputs_pred,
                                   outputs=predictions_prob,
                                   givens=givens_pred)

    def predict_batch(batch_iterator):
        preds = numpy.hstack([pred_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap) for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, _ in batch_iterator])
        return preds[:batch_iterator.n_samples]

    def predict_prob_batch(batch_iterator):
        preds = numpy.hstack([pred_prob_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap) for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, _ in batch_iterator])
        return preds[:batch_iterator.n_samples]

    train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_train, a_train, q_overlap_train, a_overlap_train, y_train], batch_size=batch_size, randomize=True)
    dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_dev, a_dev, q_overlap_dev, a_overlap_dev, y_dev], batch_size=batch_size, randomize=False)

    labels = sorted(numpy.unique(y_test))
    print 'labels', labels

    def map_score(qids, labels, preds):
        qid2cand = defaultdict(list)
        for qid, label, pred in zip(qids, labels, preds):
            qid2cand[qid].append((pred, label))

        average_precs = []
        for qid, candidates in qid2cand.iteritems():
            average_prec = 0
            running_correct_count = 0
            for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
                if label > 0:
                    running_correct_count += 1
                    average_prec += float(running_correct_count) / i
            average_precs.append(average_prec / (running_correct_count + 1e-6))
        map_score = sum(average_precs) / len(average_precs)
        return map_score

    print "Zero out dummy word:", ZEROUT_DUMMY_WORD
    if ZEROUT_DUMMY_WORD:
        W_emb_list = [w for w in params if w.name == 'W_emb']
        zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])


    def findWrongPredictions(y_true, y_pred, pids):
        wrong_pids = []
        for n, pairs in enumerate(zip(y_true, y_pred)):
            if pairs[0] != round(pairs[1]):
                print(pids[n])
                wrong_pids.append(pids[n])
        return wrong_pids

    best_dev_acc = -numpy.inf
    epoch = 0
    timer_train = time.time()
    no_best_dev_update = 0
    num_train_batches = len(train_set_iterator)

    while epoch < n_epochs:
        timer = time.time()
        for i, (x_q, x_a, x_q_overlap, x_a_overlap, y) in enumerate(tqdm(train_set_iterator), 1):
            train_fn(x_q, x_a, x_q_overlap, x_a_overlap, y)

            # Make sure the null word in the word embeddings always remains zero
            if ZEROUT_DUMMY_WORD:
                zerout_dummy_word()

            if i % 10 == 0 or i == num_train_batches:
                dev_acc = map_score(qids_dev, y_dev, predict_prob_batch(dev_set_iterator)) * 100
                # dev_acc = metrics.f1_score(y_dev, predict_batch(dev_set_iterator)) * 100
                if dev_acc > best_dev_acc:
                    print('epoch: {} batch: {} dev auc: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, i, dev_acc, best_dev_acc))
                    # wrongPreds = findWrongPredictions(y_dev, y_pred_dev, pids_dev)
                    best_dev_acc = dev_acc
                    best_params = [numpy.copy(p.get_value(borrow=True)) for p in params_test]
                    no_best_dev_update = 0
                    fname = os.path.join(nnet_outdir, 'best_dev_params_during_training')
                    cPickle.dump(best_params, open(fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

        if no_best_dev_update >= 2:
            print "Quitting after of no update of the best score on dev set", no_best_dev_update
            break

        print('epoch {} took {:.4f} seconds'.format(epoch, time.time() - timer))
        epoch += 1
        no_best_dev_update += 1

    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))


if __name__ == '__main__':
    main()
