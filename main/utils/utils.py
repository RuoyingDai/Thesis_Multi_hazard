import tensorflow as tf
import matplotlib.pyplot as plt

def R_squared(x, y, name = 'R_squared'):
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den


plt.rcParams.update({'font.size': 16})
def plot_2_metric_mean(history, name_str, p1,p2):
  #def plot_loss(history, name_str):
    #val_loss_mean = []
    plt.figure(dpi=120)
    plt.plot(history['mean_loss'], label='Training Loss',
             #color = (32/255,56/255,100/255), linewidth = 2)
             color = (225/255, 190/255, 106/255), linewidth = 2)
    plt.plot(history['mean_val_loss'], label='Validation Loss',
             #color = (125/255,125/255,191/255), linewidth = 2)
             color = (64/255, 176/255, 166/255), linewidth = 2)

    plt.title(name_str)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    plt.savefig('figures/mse_drop{}_mom{}.png'.format(p1[:5],p2[:5]),
                dpi=120,
                bbox_inches='tight')
    plt.show()
    #plt.grid(True)
    #plot_loss(history, 'LR {}/L1 Reg {}'.format(0.02,  10e-3))
    plt.figure(dpi=120)
    plt.plot(history['mean_R_squared'], label='Training R Squared',
             color = (32/255,56/255,100/255),  linewidth = 2)
    plt.plot(history['mean_val_R_squared'], label='Validation R Squared',
             color = (136/255,204/255,238/255), linewidth = 2)
    #plt.ylim([0, 10])
    plt.title(name_str)
    plt.xlabel('Epoch')
    plt.ylabel('Coefficient of Determination')
    plt.legend()
    plt.savefig('figures/r2_drop{}_mom{}.png'.format(p1[:5],p2[:5]),
                dpi=120,
                bbox_inches='tight')
    plt.show()


def plot_history_dict(drop_out_rate, momentum,
                      history_list, num_run):
    # hyperparameter 1
    p1 = str(drop_out_rate).replace(".", "_" )# drop out rate
    # hyperparameter 2
    p2 = str(momentum).replace(".", "_")# momentum
    keys0 = history_list[0].keys()
    keys = [key for key in keys0]
    key_mean_series = {}

    for key in keys:
        # possible problem: cannot calculate the mean
        key_mean_series['mean_'+key] = list(np.mean([history_list[i][key] for i in range(len(history_list))], axis = 0))
    with open('figures/mean_drop{}_mom{}'.format(p1[:5],p2[:5]), "wb") as fp:   #Pickling
        pickle.dump(key_mean_series, fp)
    # Actually plotting
    #plot_2_metric_mean(key_mean_series,
    #                   'Drop out: {}/ Momentum: {}'.format(round(drop_out_rate,4),round(momentum,4)),
    #                   p1, p2)
    # return the validation loss of the last epoch
    val_loss_mean = key_mean_series['mean_val_loss'][-1]
    print("val_loss_mean is : {}".format(round(val_loss_mean, 2)))
    return val_loss_mean
