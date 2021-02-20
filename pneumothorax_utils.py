import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime, timedelta
from time import time, strftime, gmtime

def cm_plot(cm, labels, p_size = 8, cmap = plt.cm.Reds, contrast = 4,
            subtitle = '', save_fig = False, fontweight = 'normal'):
    '''
    Plots a heatmap of the confusion matrix

    Args:
        cm: a confusion matrix array. i.e. a sklearn.metrics.confusion_matrix()
        labels: a list with class labels
        cmap: a color map
        contrast: an integer or float, higher numbers will add weight to lesser
                    values to incrase their color saturation.
        subtitle: a string to print below the title
    '''
    plot_title = '\nConfusion Matrix\n' + subtitle

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_color_vals = np.power(cm_normalized, 1/contrast)
    cm_color_vals = np.ma.masked_where(cm_color_vals == 0, cm_color_vals)
    cmap.set_under(color = 'white')

    fig = plt.figure()
    fig.set_size_inches(p_size, p_size)
    ax = fig.add_subplot(111)

    ax.imshow(cm_color_vals, cmap=cmap, interpolation='none', vmin=0.00000001)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            cell_norm = cm_normalized[x,y]
            if cell_norm > 0.99 and cell_norm < 1.: cell_norm = '> 0.99'
            elif cell_norm > 0 and cell_norm < .01: cell_norm = '< 0.01'
            else: cell_norm = format(cell_norm, '.2f')
            cell_text = '{}\n({})'.format(cm[x,y], cell_norm)
            ax.annotate(cell_text, xy=(y, x),
                    horizontalalignment='center',
                    fontweight = 'bold' if cm[x,y] > 0 else 'normal', size=12,
                    verticalalignment='center',
                    color='white' if cm_color_vals[x,y] > 0.30 else 'gray')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontweight = fontweight, size=13)
    ax.set_yticklabels(labels, fontweight = fontweight, size=13)
    ax.set_ylabel('Actual', fontweight = fontweight, size=14)
    ax.set_xlabel('Predicted', fontweight = fontweight, size=14)
    for spine in ax.spines.values():
        spine.set_edgecolor('#f1f1f1')

    title_size = min((p_size+8), 16)
    plt.title(plot_title, fontweight = 'bold', size=title_size)
    plt.tight_layout()

    if save_fig:
        file_name = 'confusion_matrix' + subtitle + '.png'
        plt.savefig(file_name, format='png', pad_inches=0.1, dpi = 480)
    plt.show()
    return None


def performance_report(DF, preds_col, neg_thresh = 0.2, pos_thresh = 0.8,
                       report_type = 'high_confidence', metrics = []):
    '''
    report_type: one of 'high_confidence', 'inconclusive', 'total'
    preds_col: (str) name of the column for predicted probabilities
    metrics: a list of strings with any of the names of the variables created
        within the function. i.e. metrics = ['accuracy', 'sensitivity']
    '''
    DF = DF[['label', preds_col]].copy()
    if report_type == 'high_confidence':
        BDF = DF[(DF[preds_col] <= neg_thresh) | (DF[preds_col] >= pos_thresh)]
    elif report_type == 'total': BDF = DF
    elif report_type == 'inconclusive':
        BDF = DF[(DF[preds_col] > neg_thresh) & (DF[preds_col] < pos_thresh)]

    BDF = BDF.assign(preds_round = np.where(BDF[preds_col] > 0.5, 1, 0))
    gt = np.array(BDF.label)
    pred = np.array(BDF.preds_round)
    accuracy = accuracy_score(gt, pred)
    total = DF.shape[0]
    determined = BDF.shape[0]
    inconclusive = total - determined
    rate_inconclusive = inconclusive/total
    TP = np.sum((gt == 1)*(pred == 1))
    TN = np.sum((gt == 0)*(pred == 0))
    FP = np.sum((gt == 0)*(pred == 1))
    FN = np.sum((gt == 1)*(pred == 0))
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)

    print('Report from {} Results\n'.format(report_type.replace('_', ' ').title()))
    print('Accuracy:          {:.2f}%'.format(accuracy*100))
    print('Sensitivity:       {:.3f}'.format(sensitivity))
    print('Specificity:       {:.3f}'.format(specificity))

    if report_type == 'high_confidence':
        print('\nTotal obs:         {}'.format(total))
        print('Predicted:         {}'.format(determined))
        print('Inconclusive:      {}'.format(inconclusive))
        print('Rate Inconclusive: {:.2f}%'.format(rate_inconclusive*100))

    elif report_type == 'inconclusive':
        print('\nInconclusive:      {}'.format(determined))

    elif report_type == 'total':
        print('\nTotal obs:         {}'.format(total))

    print('True Positives:    {}'.format(TP))
    print('True Negatives:    {}'.format(TN))
    print('False Positives:   {}'.format(FP))
    print('False Negatives:   {}\n'.format(FN))

    output_df = BDF[['label', preds_col, 'preds_round']]

    if len(metrics) > 0:
        scope = locals()
        metrics_dict = {k: eval(k) for k in metrics}
        return output_df, metrics_dict
    else: return output_df

def split_dataset(ds, steps, n_parts = 3):
    len_part = np.ceil(steps/n_parts).astype(int)
    ds_remain = ds
    ds_parts = []
    for d in range(n_parts):
        dset = ds_remain.take(len_part)
        ds_parts.append(dset)
        ds_remain = ds_remain.skip(len_part)
    return ds_parts

hline = u'\u2500'*83

def current_time_str(zone_offset = -7):
    date = datetime.today() + timedelta(hours=zone_offset)
    suffix = 'PM' if date.hour >= 12 else 'AM'
    hour = date.hour%12 if date.hour%12 else 12
    return '{:02d}:{:02d} {}'.format(hour, date.minute, suffix)

def verify_gcs_path(GCS_PATH, renew_url):
    try: tf.io.gfile.glob(GCS_PATH + '/*')[0]
    except:
        print('GCS path has expired. Follow this link to renew: ' + renew_url)
        raise Exception('GCS path has expired. Follow this link to renew: ' + renew_url) from None
    print('GCS path is valid')
    return None

# time_passed = lambda start_time: strftime("%M:%S", gmtime(time() - start_time))
def time_passed(start_time):
  secs = time() - start_time
  if secs < 3600: elapsed = strftime("%M:%S", gmtime(secs))
  elif secs < 3600*24: elapsed = strftime("%H:%M:%S", gmtime(secs))
  else: elapsed = '{:.0f} seconds'.format(secs)
  return elapsed
