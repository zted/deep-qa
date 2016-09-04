This readme illustrates how to train a nnet and make predictions for reranking, which is slightly different than
Severyn's original code.

The data used is aquaint data, but it can be any other.

First, run python parse_aquaint.py. The script looks for files named train.txt, dev.txt, test.txt, in the folder aquaint.
These files are formatted as follows:
QID - Q: This is the query .
PID 1 P: This passage corresponds to the query .
PID 0 P: I am irrelevant !
QID - Q: Here is a new query .
PID 1 P: The last one stinked anyways .
.
.
.

One that's done, a TRAIN folder should be created. Now run extract_embeddings.py

Now we're ready to train the model, which can be done by 'python run_nnet.py some.name', and during the training process
a folder in exp.out called some.name will be created, which saves the best performing parameters in each iteration.

Move the saved parameter into the folder saved_params, suppose we name it 'best.params'. Now we're ready to make a
prediction by calling 'python predict.py best.params'. The script should create a folder in exp.out with the name
best.params with the submission file in trec format in there.



A second model included here is one that also takes as input additional features. The features I use are scores that
an initial retrieval system has given to passages. This can be called using 'python run_nnet_features.py some.name'.
The code looks for extra features in files called additional_feats_dev.npy, ...train.npy. When making predictions,
run predict_features.py, it works the same way as the previous model except that it looks for additional_feats_test_retrieved.npy.
Note that there needs to be as many additional features as there are number of passages.


A third nn architecture run_nnet_dropout.py is the same as run_nnet but with attempts to add dropouts, sadly this does not work.