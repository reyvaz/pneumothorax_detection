import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime, timedelta
from time import time, strftime, gmtime
from IPython.display import display, HTML

metrics_keys = ['Accuracy', 'Sensitivity', 'Specificity', 'Total Observations',
                'Predicted', 'Correct Predictions', 'Incorrect Predictions',
                'Inconclusive', 'Rate Inconclusive',
                'True Positives', 'True Negatives', 'False Positives',
                'False Negatives']

def performance_metrics(targets, preds_prob, neg_thresh = 0.2, pos_thresh = 0.8,
                       report_type = 'high_confidence'):
    '''
    Calculates binary performance metrics
        targets: a pandas series, a list, or a 1-dim numpy array with
            ground-truth labels.
        preds_prob: a pandas series, a list, or a 1-dim numpy array with
            predicted probabilities.
        neg_thresh; (float) the lower cutoff threshold for negative predictions
        pos_thresh: (float) the upper cutoff threshold for positive predictions
            thresholds for predictions, respectively.
        report_type: (str) one of 'high_confidence', 'inconclusive', 'total'
            'total': will evaluate performance on the entire data using 0.5 as
                prediction threshold.
            'high_confidence': will evaluate performance only on the threshold
                bounded predicitons.
            'inconclusive': will return statistics on the inconclusive
                predictions according to the thresholds.

    Returns: a tuple with:
        1 dataframe with threshold bounded observations
        1 confusion matrix for the relevant observations
        1 dict with metrics for the relevant observations formated as strings

    Created to be used with cm_body_string() to create a performance report
    table on HTML.
    '''
    DF = pd.DataFrame({'label':targets, 'preds_prob':preds_prob})

    if report_type == 'high_confidence':
        BDF = DF[(DF['preds_prob'] <= neg_thresh) | (DF['preds_prob'] >= pos_thresh)]
    elif report_type == 'total': BDF = DF
    elif report_type == 'inconclusive':
        BDF = DF[(DF['preds_prob'] > neg_thresh) & (DF['preds_prob'] < pos_thresh)]

    BDF = BDF.assign(preds_round = np.where(BDF['preds_prob'] > 0.5, 1, 0))
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

    metrics_strings = {
            'Accuracy': '{:.2f}%'.format(accuracy*100),
            'Sensitivity': '{:.3f}'.format(sensitivity),
            'Specificity': '{:.3f}'.format(specificity),
            'True Positives': str(TP),
            'True Negatives': str(TN),
            'False Positives': str(FP),
            'False Negatives': str(FN),
            'Correct Predictions': str(TN + TP),
            'Incorrect Predictions': str(FN + FP),
    }

    if report_type == 'high_confidence':
        metrics_strings['Total Observations'] = str(total)
        metrics_strings['Predicted'] = str(determined)
        metrics_strings['Inconclusive'] = str(inconclusive)
        metrics_strings['Rate Inconclusive'] = '{:.2f}%'.format(rate_inconclusive*100)

    elif report_type == 'inconclusive':
        metrics_strings['Inconclusive'] = str(determined)

    elif report_type == 'total':
        metrics_strings['Total Observations'] = str(total)

    cm = confusion_matrix(BDF.label, BDF.preds_round)
    return BDF, cm, metrics_strings

def metrics_table_html_string(metrics_strings):
    '''
    Used the output of performance_metrics() to create the code for a performance
    metrics table in HTML.
        metrics_strings: (dict) with metrics names as keys and values as strings
    '''
    mkeys = [k for k in metrics_keys if k in metrics_strings]
    table_string = '<table>\n   <tbody>\n'
    for k in mkeys:
        if k == 'Total Observations' or k == 'True Positives':
            table_string += '\t<tr><td>&emsp;</td><td></td></tr>\n'
        table_string += '\t<tr><td>{}&emsp;</td><td class="idented">{}</td></tr>\n'.format(k, metrics_strings[k])
    table_string += '   </tbody>\n</table>'
    return table_string

def cm_html_list(cm, max_rgb = (0, 45, 66), contrast = 0.7, blank_zeros = False,
                 text_color_thresh = 0.2):
    '''
    Calculates cell values to populate a confusion matrix in HTML
        cm: (numpy array) a (squared) confusion matrix
        max_rgb: (tuple of ints) with the rgb values for the darkest possible
            cell in the CM.
        contrast: (float) adjust to tweak the contrast between cells

    Returns a list of lenght cm.shape[0] containing lists of strings to be used
        by cm_body_string() to populate CM cells
    '''
    mc = np.array(max_rgb)
    min_rgb = (239, 239, 239)
    mn = np.array(min_rgb)
    dist = mn - mc
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    str_list = []
    for row_v, row_n in zip(cm, cm_normalized):
        row_list = []
        for v, n in zip(row_v, row_n):
            if v == 0: bg_color = (239, 239, 239)
            else: bg_color = tuple((mc+(dist*(1.-n**contrast))).astype(int))
            if n > text_color_thresh: txt_color = 'white'
            else: txt_color = 'grey'
            if n > 0.99 and n < 1.: n = '(> 0.99)'
            elif n > 0 and n < .01: n = '(< 0.01)'
            elif v == 0 and blank_zeros: n = ''; v = ''
            else: n = '({:.2f})'.format(n)
            n = '<span class="norm">{}</span>'.format(n)
            str_cell = [str(bg_color), txt_color] + [str(v), n]
            row_list += str_cell
        str_list.append(row_list)
    return str_list

# html base for 2 columns
html_report_base = '''
<style>
.column {float: left; width: 320px; padding: 10px;}
.idented {text-indent: 1em;}
</style>
<h3>Performance Report</h3>
<div>
    <div class="column"> <div class="cm"> %s </div> </div>
    <div class="column"> <div class="cm"> %s </div> </div>
</div>
'''

cm_css = '''
.cm {height: 250px; display:table-cell; vertical-align:middle;}
.tg {border-spacing:0;text-align:center; vertical-align:middle}
.tg td {overflow:hidden; padding:8px 6px; word-break:normal;}
.tg th {overflow:hidden; word-break:normal;}
.ver_text {transform: rotate(-90deg);}
.norm {font-size:11.5px}'''

cm_header = '''
<table class="tg">
<thead>
    <tr>
      <th class="tg" colspan="2" rowspan="2"></th>
      <th class="tg" colspan="%s">Confusion Matrix<br><span class="norm">%s</span></th>
    </tr>
    <tr><td class="tg" colspan="%s">Predicted</td></tr>
</thead>
<tbody>
    '''
cm_footer = '''
    <tr>
      <td class="tg"></td>
      <td class="tg"></td>
      %s
    </tr>
</tbody>
</table>
    '''

# base for one cell of cm values
cm_cell_base = '<td class="tg" style="background-color:rgb%s; color:%s">%s<br>%s</td>'

def confusion_matrix_html(cm, classes = 'default', show_cm = True, title = '',
                          max_rgb = (0, 45, 65), contrast = 0.7,
                          blank_zeros = False, text_color_thresh = 0.2,
                          incl_css = True, break_labels = True,
                          html_file = False):
    '''
    Creates a string with the HTML code for a confusion matrix
    Args:
        cm: (numpy array) a square array confusion matrix
        classes: 'default' or a list of classes
        show_cm: (bool) if True, it will display the confusion matrix
            it requires `display` and `HTML` from `IPython.display`
        title: (str)
        max_rgb: (tuple of ints) with the rgb values for the darkest possible
            cell in the confusion matrix. Recommended at least 2 values to be
            less than 100
        contrast: (float) adjust to tweak the contrast between cells
        blank_zeros: (bool) if True, it will leave blank cells with 0s
        text_color_thresh: (float) adjust to improve visibility of text in
            colored cells
        incl_css: if True, the output string will include additional CSS code
            for formatting confusion matrix cells.
        break_labels: (bool) if True, it will break class names at spaces
        html_file: False or str. If str, it will save the html code to a file.
            if no extension is provided it defaults to .html.

    Returns: a string containing the full html code for a confusion matrix
    '''
    cell_vals = cm_html_list(cm, max_rgb, contrast, blank_zeros, text_color_thresh)
    if classes == 'default':
        classes = ['Class {}'.format(i+1) for i in range(cm.shape[0])]

    if break_labels: classes = [x.replace(' ', '<br>') for x in classes]
    n_classes = len(classes)

    if incl_css: header = '<style>{}\n</style>{}'.format(cm_css, cm_header)
    else: header = cm_header
    header = header % (n_classes, title, n_classes)

    row_string_for_data = cm_cell_base*n_classes

    data_string = ''
    for i, (c, row) in enumerate(zip(classes, cell_vals)):
        start_row = '<tr>\n'
        if i == 0:
            start_row += '\t<td class="tg ver_text" rowspan="%s">Actual</td>\n' % n_classes
        fill_row = '\t<td class="tg">%s</td>%s</tr>' % (c, row_string_for_data)
        fill_row = fill_row % tuple(row)

        data_string += start_row + fill_row

    footer_classes = '<td class="tg" style="width: 60px;">%s</td>'*n_classes
    footer =  cm_footer % footer_classes
    footer =  footer % tuple(classes)
    cm_string = header + data_string + footer
    if html_file:
        assert isinstance(html_file, str), 'html_file var must be False or a string'
        if '.' not in html_file: html_file = html_file + '.html'
        with open(html_file, 'w') as H: H.write(cm_string)
    if show_cm: display(HTML(cm_string))
    return cm_string

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

def time_passed(start_time):
  secs = time() - start_time
  if secs < 3600: elapsed = strftime("%M:%S", gmtime(secs))
  elif secs < 3600*24: elapsed = strftime("%H:%M:%S", gmtime(secs))
  else: elapsed = '{:.0f} seconds'.format(secs)
  return elapsed
