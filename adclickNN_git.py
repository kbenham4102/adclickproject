import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import adclickimport as imp


def batch_norm_layer(X, units, training = True, regularizer = None,
                    act_fn = 'relu'):

    step1 = tf.contrib.layers.fully_connected(X, units, activation_fn = None,
                                            weights_regularizer = regularizer)

    step2 = tf.contrib.layers.batch_norm(step1, center = True, scale = True,
                                                is_training = training)

    if act_fn == 'relu':
        return tf.nn.relu(step2)
    else:
        return step2



def forward_propagation(X, params, batchnorm = False, regularizer = None,
training = True):

    net = X
    initializer = tf.contrib.layers.xavier_initializer(uniform = False)
# regularizer syntax = tf.layers.l2_regularizer(scale = 0.1)
    if batchnorm == False:
        for units in params['hidden_units'][:-1]:
            net = tf.layers.dense(net, units = units, activation = tf.nn.relu,
            kernel_initializer = initializer,
            kernel_regularizer = regularizer)

        out_units = params['hidden_units'][-1]
        logits = tf.layers.dense(net, out_units, activation = None,
        kernel_initializer = initializer,
        kernel_regularizer = regularizer)
    elif batchnorm == True:
        for units in params['hidden_units'][:-1]:
            net = batch_norm_layer(net, units, training = training,
                                    regularizer = regularizer, act_fn = 'relu')
        out_units = params['hidden_units'][-1]
        logits = batch_norm_layer(net, out_units , training = training,
                                      regularizer = regularizer, act_fn = None)
    return logits



def cost_fn(logits, Y, regularizer = None, pos_wt = 1):

    cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                    targets = Y, logits = logits, pos_weight = pos_wt))

    if regularizer != None:
        reg_term = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = cross_entropy + tf.reduce_sum(reg_term)
    else:
        loss = cross_entropy


    return loss

def model_fn(params, batchnorm = False, regularizer = None, pos_wt = 1,
lrdecay = False):
    # Initialize placeholders
    X = tf.placeholder(tf.float32, shape = (None, params['num_features']))
    Y = tf.placeholder(tf.float32, shape = (None, 1))

    # Compute the final linear step of output layer, includes varibale
    # initializers in tf.layers functions
    logits = forward_propagation(X, params, batchnorm = batchnorm,
    regularizer = regularizer, training = True)

    # Calculate cost w/class wt options from sigmoid function
    # logits are transformed by output activation included with tf function
    cost = cost_fn(logits, Y, regularizer = regularizer, pos_wt = pos_wt)

    # Define optimizer with option for LR decay
    global_step = tf.Variable(0, trainable = False)
    if lrdecay == True:
            starter_lrt = params['learn_rt']
            learning_rate = tf.train.exponential_decay(starter_lrt,
                                    global_step, 100, 0.96, staircase = True)
    else:
        learning_rate = params['learn_rt']
        # Adam optimization
    optimizer = tf.train.AdamOptimizer(
    learning_rate = learning_rate).minimize(cost, global_step = global_step)



    num_mb = params['num_minibatches']
    learn_curve = []

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver(filename = params['modelname'])
    with tf.Session() as sess:
        sess.run(init_op)

        for epoch in range(params['num_epochs']):
            epoch_cost = 0
            for n in range(num_mb - 1):
                # Obtain minibatch training set from master training set.
                # Minibatches have been pre-allocated in hard memory
                # Prepared by taking a random ~50,000,000 batch from master
                # set, shuffling, centering, scaling, and dividing into 25,001
                # example mini-batches.
                X_trainMB, Y_trainMB = imp.getbatch(params['batchlib'],
                                                    'MBcent' + str(n) + '.csv')

                # Determine cost and update model parameters based on minibatch
                _, mb_cost = sess.run([optimizer, cost], feed_dict =
                {X:X_trainMB, Y:Y_trainMB})

                epoch_cost += mb_cost/num_mb

                if n%10 == 0:
                    learn_curve.append(epoch_cost)
                if n%100 == 0 and params['print_cost'] == True:
                    print(epoch_cost)
                if n == 1998:
                    saver.save(sess, params['save_path'])



            if params['print_cost'] == True:
                print('Cost for epoch ' + str(epoch + 1) + ': ', epoch_cost)

        # Define basic accuracy protocol
        prediction = tf.cast(tf.greater(tf.sigmoid(logits), params['positive_thresh']),
        tf.float32)
        eval_predict = tf.equal(prediction, Y)
        accuracy = tf.reduce_mean(tf.cast(eval_predict, "float"))
        roc, update_op = tf.metrics.auc(Y, tf.sigmoid(logits))


        # Run accuracy protocol on test set
        sess.run(tf.local_variables_initializer())
        X_dev, Y_dev = imp.getbatch(params['batchlib'], 'MBcent1998.csv')
        print('Dev Set Accuracy: ', accuracy.eval({X: X_dev, Y: Y_dev}))

        print('Area under ROC Curve: {}'.format(sess.run([roc, update_op],
               feed_dict = {Y:Y_dev, X:X_dev})))

        if params['runtestset'] == True:
            result = pd.DataFrame()
            X_test = imp.get_testset()
            predict = tf.sigmoid(logits)
            P = sess.run(predict, feed_dict = {X:X_test})
            result['click_id'] = np.arange(len(P))
            result['is_attributed'] = P
            result.to_csv(params['save_path'] + 'testpred.csv', index = False)

    return None



params = {}

units_per_layer = [128, 64, 32, 16, 1]
poswt = 454 # Imbalanced dataset
params['hidden_units'] = units_per_layer
params['num_features'] = 10
params['learn_rt'] = 0.001
params['num_epochs'] = 30
params['num_minibatches'] = 2000

params['print_cost'] = True
params['positive_thresh'] = 0.95
params['batchlib'] = 'C:/Users/Kevin/scripts/adclickproj/MBcent'
params['modelname'] = 'Model5_reg0001_poswt_' + str(poswt)
params['save_path'] = 'C:/Users/Kevin/scripts/adclickproj/Models/' + params['modelname'] + '/'

params['runtestset'] = False
# typical regularizer = tf.contrib.layers.l2_regularizer(scale = 0.001)
print("Running the model...")
_ = model_fn(params, batchnorm = False,
regularizer = None, pos_wt = poswt)
