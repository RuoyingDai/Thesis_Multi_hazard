import tensorflow as tf
from .utils import R_squared
import numpy as np

def my_model(learning_rate, regularization,layer, X_train, y_train):
    num_run = 1
    history = [[] for i in range(num_run)]
    val_metric = 0
    for run_idx  in range(num_run):
        print('Same setting run: {}'.format(run_idx + 1))
        history_run = my_model0(learning_rate, regularization,layer, run_idx, X_train, y_train)
        history[run_idx] = history_run
        val_metric += history_run['val_loss'][-1]
    p1 = str(learning_rate).replace(".", "_" )# learning rate
    # hyperparameter 2
    p2 = str(regularization).replace(".", "_")# regularization
    p3 = round(layer) # activation function
    with open('/Users/pika/Documents/multihazard/bo_nn/adagrad_lr{}_reg{}_lay{}'.format(p1[:5],p2[:5],p3), "wb") as fp:   #Pickling
        pickle.dump(history, fp)
    #val_metric = plot_history_dict(drop_out_rate, momentum,
    #                  history, num_run)
    #return val_metric
    return -round(val_metric/num_run,5)


def my_model0(lr, reg, lay,run_idx, X_train, y_train):
    lay_idx =round(lay)
    print(lay_idx)
    lay2 =['tanh', 'relu', 'sigmoid']
    lay3 = ['relu', 'sigmoid', 'tanh']
    lay4 = ['sigmoid', 'tanh', 'relu']
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu',
                            activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-2, l2=reg*1e-1)),
      #tf.keras.layers.Dense(int(hp_neuron1), activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(256, activation=lay2[lay_idx],
                            activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-1, l2=reg*1e-0)),
      #tf.keras.layers.Dense(int(hp_neuron2), activation='tanh',activity_regularizer=regularizers.l1(hp_reg)),
      tf.keras.layers.Dense(256, activation=lay3[lay_idx],
                            activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-1, l2=reg*1e-0)),
      #tf.keras.layers.Dense(int(hp_neuron3), activation='sigmoid'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(128, activation=lay4[lay_idx],
                            activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-2, l2=reg*1e-1)),
      tf.keras.layers.Dense(64, activation = 'tanh',
                            activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-2, l2=reg*1e-0)),
      tf.keras.layers.Dense(32)
  ])

    #Compile the model
    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),# it does not work if () is not here.
        #optimizer = tf.keras.optimizers.SGD(learning_rate= 0.005,
        #                                    momentum = momentum),# 0.05/0.02
        optimizer = tf.keras.optimizers.Adagrad(
          learning_rate = lr,
          #learning_rate=0.001,# default is 0.001
          initial_accumulator_value=0.1,
          epsilon=1e-07,
          name='Adagrad',
          ),
        metrics = [
            R_squared,
        ]
    )

    overfitCallback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                   min_delta=0.00001,
                                                   patience = 500,
                                                   mode = 'min')
    history = model.fit(np.array(X_train), np.array(y_train),
            batch_size=5,
            epochs=20,
            verbose=True,
            #verbose =False,
            validation_split = 0.2,
    callbacks=[overfitCallback])
    model.predict(X_test, y_test)
            #callbacks=[tensorboard_callback])
    # return the MSE of the last update of models
    #print(-history.history['val_loss'][-1])
    #return -history.history['val_loss'][-1]
    return history.history
