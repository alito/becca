import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

""" 
Constants and functions for used across the BECCA core
"""
# Shared constants
EPSILON = sys.float_info.epsilon
BIG = 10 ** 20
MAX_INT16 = np.iinfo(np.int16).max

DARK_GREY = (0.2, 0.2, 0.2)
LIGHT_GREY = (0.9, 0.9, 0.9)
RED = (0.9, 0.3, 0.3)
# BECCA pallette
COPPER_HIGHLIGHT = (253./255., 249./255., 240./255.)
LIGHT_COPPER = (242./255., 166./255., 108./255.)
COPPER = (175./255., 102./255, 53./255.)
DARK_COPPER = (132./255., 73./255., 36./255.)
COPPER_SHADOW = (25./255., 22./255, 20./255.)
OXIDE = (20./255., 120./255., 150./255.)

def weighted_average(values, weights):
    """ Perform a weighted average of values, using weights """
    weighted_sum_values = np.sum(values * weights, axis=0) 
    sum_of_weights = np.sum(weights, axis=0) 
    return (weighted_sum_values / (sum_of_weights + EPSILON))[:,np.newaxis]

def generalized_mean(values, weights, exponent):
    shifted_values = values + 1.
    values_to_power = shifted_values ** exponent
    mean_values_to_power = weighted_average(values_to_power, weights)
    shifted_mean = (mean_values_to_power + EPSILON) ** (1./exponent)
    mean = shifted_mean - 1.
    # Find means for which all weights are zero. These are undefined.
    # Set them equal to zero.
    sum_weights = np.sum(weights, axis=0)
    zero_indices = np.where(np.abs(sum_weights) < EPSILON)
    mean[zero_indices] = 0.
    return mean

def map_one_to_inf(a):
    """ Map values from [0, 1] onto [0, inf) and map values 
    from [-1, 0] onto (-inf, 0] """
    eps = np.finfo(np.double).eps
    a_prime = np.sign(a) / (1 - np.abs(a) + eps) - np.sign(a)
    return a_prime

def map_inf_to_one(a_prime):
    """ Map values from [0, inf) onto [0, 1] and map values 
    from  (-inf, 0] onto [-1, 0] """
    a = np.sign(a_prime) * (1 - 1 / (np.abs(a_prime) + 1))
    return a

def bounded_sum(a, axis=0):
    """ 
    Sum elements nonlinearly, such that the total is less than 1 
    
    To be more precise, as long as all elements in a are between -1
    and 1, their sum will also be between -1 and 1. a can be a 
    list or a numpy array. 
    """ 
    if type(a) is list:
        total = map_one_to_inf(a[0])
        for item in a[1:]:
            total += map_one_to_inf(item)
        return map_inf_to_one(total)
    else:
        # handle the case where a is a one-dimensional array
        if len(a.shape) == 1:
            a = a[:, np.newaxis]
        bounded_total = map_inf_to_one(np.sum(map_one_to_inf(a), axis=axis))
        return bounded_total[:,np.newaxis]

def pad(a, shape, val=0.):
    """
    Pad a numpy array to the specified shape
    
    If any element of shape is 0, that size remains unchanged in 
    that axis. If any element of shape is < 0, the size in that
    axis is incremented by the magnitude of that value.
    Use val (default 0) to fill in the extra spaces. 
    """
    if shape[0] <= 0:
        rows = a.shape[0] - shape[0]
    else:
        rows = shape[0]
        # debug
        if rows < a.shape[0]:
            print ' '.join(['a.shape[0] is', str(a.shape[0]), ' but trying to',
                            ' pad to ', str(rows), 'rows.'])
    if shape[1] <= 0:
        cols = a.shape[1] - shape[1]
    else:
        cols = shape[1]
        # debug
        if cols < a.shape[1]:
            print ' '.join(['a.shape[1] is', str(a.shape[1]), ' but trying to',
                            ' pad to ', str(cols), 'cols.'])
    padded = np.ones((rows,cols)) * val
    padded[:a.shape[0], :a.shape[1]] = a
    return padded

def str_to_int(exp):
    """ Convert a string to an integer """ 
    sum = 0
    for character in exp:
        sum += ord(character)
    return sum

def visualize_array(image_data, shape=None, save_eps=False, 
                    label='data_figure', epsfilename=None):
    """ Produce a visual representation of the image_data matrix """    
    if shape is None:
        shape = image_data.shape
    if epsfilename is None:
        epsfilename = 'log/' + label + '.eps'
    fig = plt.figure(str_to_int(label))
    
    # Diane made the brilliant suggestion to leave this plot in color. 
    # It looks much prettier.
    plt.summer()
    im = plt.imshow(image_data[0:shape[0], 0:shape[1]])
    im.set_interpolation('nearest')
    plt.title(label)
    fig.show()
    fig.canvas.draw()
    if save_eps:
        fig.savefig(epsfilename, format='eps')
    return

def get_files_with_suffix(dir_name, suffixes):
    """ Get all of the files with a given suffix in dir recursively """
    found_filenames = []
    for localpath, directories, filenames in os.walk(dir_name):
        for filename in filenames:
            for suffix in suffixes:
                if filename.endswith(suffix):
                    found_filenames.append(os.path.join(localpath, filename))
    found_filenames.sort()
    return found_filenames

def report_roc(ground_truth_filename, surprise_log_filename, self_name):
    """
    Report the Receiver Operating Characteristic curve

    Plot the true positive rate (the number of correctly identified
    targets divided by the total number of targets) against the
    false positive rate (the number of data points erroneously 
    identified as targets divided by the total number of 
    non-target data points).
    """
    truth = np.loadtxt(ground_truth_filename)
    surprise = np.loadtxt(surprise_log_filename)
    abs_surprise = np.abs(surprise[:,0])
    times = surprise[:,1]
    # If a target is identified within delta seconds, that is close enough
    delta = 0.#1
    # Include a time delay (seconds) to account for the time it takes for 
    # information to move up levels in the hierarchy
    delay = 0.#2
    starts = truth[:,0] - delta + delay
    ends = truth[:,1] + delta + delay
    total_num_targets = starts.size
    # Total up potential false positives.
    total_non_target_points = 0
    for time in times:
        after_start = np.where(time > starts, True, False)
        before_end = np.where(time < ends, True, False)
        target_match = np.logical_and(after_start, before_end)
        if not target_match.any():
            total_non_target_points += 1

    false_positive_rate = []
    true_positive_rate = []
    thresholds = np.linspace(np.min(abs_surprise), np.max(abs_surprise), 
                             num=100)
    for threshold in thresholds:
        # Determine the false positive rate, i.e. how many
        # of all possible false positives were reported
        above_threshold_indices = np.where(abs_surprise > threshold)
        above_threshold_times = times[above_threshold_indices]
        num_false_positives = 0
        for time in above_threshold_times:
            after_start = np.where(time > starts, True, False)
            before_end = np.where(time < ends, True, False)
            target_match = np.logical_and(after_start, before_end)
            if not target_match.any():
                num_false_positives += 1
        false_positive_rate.append(float(num_false_positives) /
                                   (float(total_non_target_points) +
                                    EPSILON))
        # Determine the true positive rate, i.e.
        # what fraction of the targets were identified 
        num_targets_identified = 0
        for indx in range(total_num_targets):
            after_start = np.where(times[above_threshold_indices] > 
                                   starts[indx], True, False)
            before_end = np.where(times[above_threshold_indices] < 
                                  ends[indx], True, False)
            target_match = np.logical_and(after_start, before_end)
            if target_match.any():
                num_targets_identified += 1
        true_positive_rate.append(float(num_targets_identified)/
                                          (float(total_num_targets) + 
                                           EPSILON))
    # Calculate a single number to characterize the ROC curve:
    # the area under the curve 
    num_points = 1000
    fpr_fine = np.linspace(0., 1., num=num_points)
    tpr_fine = np.interp(fpr_fine, np.array(false_positive_rate)[::-1], 
                         np.array(true_positive_rate)[::-1], 
                         left=0., right=1.)
    roc_integral = np.sum(tpr_fine) / num_points

    # Show surprise over time 
    fig = plt.figure(str_to_int('surprise'))
    fig.clf()
    plt.plot(times, abs_surprise)
    plt.title('Novel target identification signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Estimated novelty strength')
    plt.hold(True)
    ax = plt.gca()
    # Show the temporal locations of the targets
    for target_index in range(total_num_targets):
        ax.add_patch(mpatches.Rectangle(
                (starts[target_index], np.min(abs_surprise)), 
                ends[target_index] - starts[target_index], 
                np.max(abs_surprise) - np.min(abs_surprise), 
                facecolor=LIGHT_COPPER, edgecolor=COPPER_SHADOW))
    # Save the surprise history plot
    filename =  ''.join((self_name, '_novelty_vs_targets.png'))
    full_filename = os.path.join('log', filename)
    plt.savefig(full_filename, format='png') 
        
    # Show the ROC curve
    fig = plt.figure(str_to_int('roc'))
    fig.clf()
    plt.plot(false_positive_rate, true_positive_rate)
    title_text = ''.join(('Receiver operating characteristic (ROC)',
                          ' curve for ', self_name))
    body_text = ''.join(('Area under the ROC curve = ', 
                         '%0.3f' % roc_integral))
    plt.title(title_text)
    plt.text(0.4, 0.2, body_text, color=COPPER_SHADOW, size=10)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.axis((-0.1, 1.1, -0.1, 1.1))
    # Save the ROC plot
    filename =  ''.join((self_name, '_roc.png'))
    full_filename = os.path.join('log', filename)
    plt.savefig(full_filename, format='png') 
    plt.ioff()
    plt.show()    
    return roc_integral
