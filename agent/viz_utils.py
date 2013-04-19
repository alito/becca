
import copy 
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import utils 

""" A set of methods for visualizing aspects of the BECCA's internal state and operations """

def visualize_array(image_data, shape=None, save_eps=False, label='data_figure', epsfilename=None):
    """ Produce a visual representation of the image_data matrix """    
    if shape is None:
        shape = image_data.shape
    if epsfilename is None:
        epsfilename = 'log/' + label + '.eps'
    fig = plt.figure(utils.ord_str(label))
    
    """ Diane L. made the brilliant suggestion to leave this plot in color. It looks much prettier. """
    plt.summer()
    im = plt.imshow(image_data[0:shape[0], 0:shape[1]])
    im.set_interpolation('nearest')
    plt.title(label)
    fig.show()
    fig.canvas.draw()
    if save_eps:
        fig.savefig(epsfilename, format='eps')
    return
        
def reduce_feature_set(perceiver, n_primitives, n_actions):
    """ Take in the feature map and express each feature in terms of the lowest level inputs 
    (sensors, primitives, and actions) that excite them. Each row of the feature map represents 
    a single feature's composition. Expand each row until it is represented only in terms of low level
    inputs. Start at the highest non-zero column and work from right to left.  
    """
    
    """ The highest nonzero column will not be more than the number of sensors plus the number 
    of features, which includes primitives, actions, and created features, modified to account for 
    python's indexing and range behavior.
    """
    first_feature_index = perceiver.num_sensors + perceiver.num_features -1
    
    """ The last feature to be expanded will be the first created feature, i.e. the number of sensors plus 
    the number of primitives and actions, modified to account for python's indexing and range behavior.
    """
    last_feature_index = perceiver.num_sensors + n_primitives + n_actions - 1
    reduced_feature_set = copy.deepcopy(perceiver.feature_map[:perceiver.num_features,
                                                              :first_feature_index + 1])
    
    for column_index in range(first_feature_index, last_feature_index, -1):
        features_that_have_this_feature_as_input = np.nonzero(reduced_feature_set[:,column_index])[0]
        
        if features_that_have_this_feature_as_input.size > 0:
            input_magnitudes = reduced_feature_set[features_that_have_this_feature_as_input, column_index]
            feature_index = column_index - perceiver.num_sensors
            
            """ Add the contribution from the features in terms of its lower level representation """
            this_feature = reduced_feature_set[feature_index,:]
            reduced_feature_set[features_that_have_this_feature_as_input,:] += \
                np.dot(input_magnitudes[:,np.newaxis], this_feature[np.newaxis,:], )
    
            """ Remove the high level representation, just to clean up """
            reduced_feature_set[features_that_have_this_feature_as_input, column_index] = 0
            
    return reduced_feature_set[n_primitives + n_actions:,:perceiver.num_sensors]

    
def visualize_model(actor, num_primitives, num_actions, n=None):
    """ Visualize some of the transitions in the model """
    if n == None:
        n = actor.num_transitions
    n = np.minimum(n, actor.num_transitions)
        
    print "The actor has a total of ", actor.num_transitions, " transitions."
    
    """ Show the n transitions from the actor that have the highest impact.
    NOTE: argsort returns indices of sort in *ascending* order. 
    """
    index_by_rank = np.argsort(actor.reward_value * (np.log(actor.count + 1)), axis=1).ravel()
    
    for index in range(n):
        print "Showing the " + str(index) + "th most impact."            
        visualize_transition(actor, num_primitives, num_actions, index_by_rank[-(index+1)])
        plt.show()
    return
    
        
def visualize_transition(actor, num_primitives, num_actions, transition_index, save_eps=False, 
                          label=None, epsfilename='log/transition.eps'):
    """ Visualize a single model transition """
    if label==None:
        label = 'Transition ' + str(transition_index)
    n_features = actor.num_features
        
    fig = plt.figure(utils.ord_str(label))
    fig.clf()
    plt.ioff()    
    viz_axes = fig.add_subplot(1,1,1)

    
    count = actor.count[0, transition_index]
    reward_value = actor.reward_value[0, transition_index]
    reward_uncertainty = actor.reward_uncertainty[0, transition_index]
    goal_value = actor.goal_value[0, transition_index]
    
    plt.title('Transition {:}'.format(transition_index) + 
              '  count: {:.2f}'.format(count)  + 
              '  reward value: {:.2f}'.format(reward_value) + 
              '  uncertainty: {:.2f}'.format(reward_uncertainty) + 
              '  goal value: {:.2f}'.format(goal_value) )
    plt.xlabel(label)
    
    context = np.copy(actor.context[:n_features, transition_index, np.newaxis])
    cause = np.copy(actor.cause[:n_features, transition_index, np.newaxis])
    effect = np.copy(actor.effect[:n_features, transition_index, np.newaxis])
    effect_uncertainty = np.copy(actor.effect_uncertainty[:n_features, transition_index, np.newaxis])

    visualize_state(context, num_primitives, num_actions, y_max=3.75, y_min=3.25, axes=viz_axes)
    visualize_state(cause, num_primitives, num_actions,  y_max=2.75, y_min=2.25, axes=viz_axes)
    visualize_state(effect, num_primitives, num_actions,  y_max=1.75, y_min=1.25, axes=viz_axes)
    visualize_state(effect_uncertainty, num_primitives, num_actions, y_max=0.75, y_min=0.25, axes=viz_axes)
    
    """ This trick makes matplotlib recognize that it has something to plot.
    Everything else in the plot is patches and text, and for some reason
    it doesn't draw unless you include this plot command.
    """ 
    plt.plot(0, 0, color='black') 
    
    fig.show()
    fig.canvas.draw()
    if save_eps:
        fig.savefig(epsfilename, format='eps')
    return
  
  
def visualize_state(state, num_primitives, num_actions, label='state', y_min=0.25, y_max=0.75, 
                      save_eps=False, epsfilename='log/state.eps', axes=None):
    """ Present the state in a visually intuitive way. height_proportion is the height of the display as 
    a fraction of width.
    """       
    if axes == None:
        fig = plt.figure(utils.ord_str(label))
        fig.clf()
        fig.set_facecolor((0.8, 0.8, 0.8))
        axes = fig.add_subplot(1,1,1)
        plt.title(label)
        axes.set_ylim(0.0, 1.0)
     
    """ The number of blank features in between groups """
    x_spacer_width = 3 
    total_width = 0

    primitives = state[:num_primitives,:]
    actions = state[num_primitives:num_primitives + num_actions,:]
    features = state[num_primitives + num_actions:,:]
    
    total_width += primitives.size
    total_width += x_spacer_width
    total_width += actions.size
    total_width += x_spacer_width
    total_width += features.size
    
    axes.set_xlim(-x_spacer_width, total_width + x_spacer_width)
        
    x = 0
    for indx in range(primitives.size):
        rectPatch(x, x + 1, y_min, y_max, primitives[indx], axes)
        x += 1
        
    x += x_spacer_width
    for indx in range(actions.size):
        rectPatch(x, x + 1, y_min, y_max, actions[indx], axes)
        x += 1
    
    x += x_spacer_width
    for indx in range(features.size):
        rectPatch(x, x + 1, y_min, y_max, features[indx], axes)
        x += 1
    
    """ Get the figure to recognize that it has something to plot. A blatant hack.  """       
    plt.plot(0, 0, color='black') 

    if save_eps:
        fig.savefig(epsfilename, format='eps')
    return


def rectPatch (left_x, right_x, lower_y, upper_y, value=0.5, 
           axes=plt.gca(), borders=False):
    """ Draw a rectangular patch bounded by (left_x, lower_y) on the lower, left hand corner and 
    (right_x, upper_y) on the upper, right hand corner. The grayscale value of the patch can 
    be specified, as can the axes in which to draw the patch.
    """
    verts = [
            (left_x, lower_y), # left, bottom
            (left_x, upper_y), # left, top
            (right_x, upper_y), # right, top
            (right_x, lower_y), # right, bottom
            (left_x, lower_y), # ignored
            ]
    
    codes = [Path.MOVETO,
             Path.LINETO,
               Path.LINETO,
               Path.LINETO,
               Path.CLOSEPOLY,
               ]
    
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=[value, value, value], lw=0)
    axes.add_patch(patch)
    if borders:
        for i in range(1):
            plt.plot((verts[i][0], verts[i+1][0]), 
                   (verts[i][1], verts[i+1][1]), color='black')
    return
    
    
    '''def force_redraw():
    """ Force matplotlib to draw things on the screen. Pause is needed for events to be processed. 
    Qt backend needs two event rounds to process screen. Any number > 0.01 and <=0.02 would do.
    """
    plt.pause(0.015)
    '''
