import copy 
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import state
import utils

""" A set of methods for visualizing aspects of the Becca's internal state and
operations. Not for visualizing world-specific information, or 
world-specific interpretation of Becca's information. That should be 
taken care of by individual worlds.  
"""

def visualize_coactivity(coactivity, size=0, 
                                    save_eps=False, 
                                    epsfilename='log/coactivity.eps'):
    """ Produce a visual representation of the coactivity matrix """
    
    if size == 0:
        size = coactivity.shape[0]
    fig = plt.figure("perceiver coactivity visualization")
    
    """ Diane L. made the brilliant suggestion to leave this plot in color. 
    It looks much prettier.
    """
    plt.summer()

    im = plt.imshow(coactivity[0:size, 0:size])
    im.set_interpolation('nearest')
    plt.title("Coactivity among inputs")
    plt.draw()
    
    if save_eps:
        fig.savefig(epsfilename, format='eps')
        
    return
  
 
def visualize_feature_map(feature_map):
    plt.figure("feature map visualization")
    
    """ Diane L. made the brilliant suggestion to leave this plot in color. 
    It looks much prettier.
    """
    plt.summer()
    im = plt.imshow(feature_map)
    im.set_interpolation('nearest')
    plt.title("Feature map")
    plt.draw()
        
 
def visualize_grouper_hierarchy(perceiver, save_eps=False, 
                                  epsfilename='log/hierarchy.eps'):
    """ Produce a visual representation of the feature group 
    inheritance hierarchy. 
    """
    
    """ Radial distance of the text labels from the center of the 
    circle, assuming a radius of 1.0
    """
    text_distance = 1.2
    
    """ Assign x,y locations for the nodes representing each 
    feature group. A circular arrangement gives every node
    a line-of-sight to all the other nodes. This is appropriate since
    any feature group may have inputs coming from any lower-numbered 
    feature group.
    """
    n_nodes = perceiver.grouping_map_group.n_feature_groups() + 3
    delta_angle = 2. * np.pi / float(n_nodes)
    nodes = np.zeros((n_nodes,2))
    node_text = []
    
    """ Start at South, and work clockwise around the circle in 
    equal increments.
    """
    for node_index in range(n_nodes):
        angle = delta_angle * float(node_index)
        nodes[node_index,0] = -np.sin(angle)
        nodes[node_index,1] = -np.cos(angle)
        """ node n represents feature group n - 3 """
        node_text.append(str(node_index - 3))   
        
    """ node 0 represents the sensors group """
    node_text[0] = 's'  
    """ node 1 represents the primitives group """
    node_text[1] = 'p'  
    """ node 2 represents the action group """
    node_text[2] = 'a'  
    
    """ Display the nodes """   
    """ Prepare the plot """         
    fig = plt.figure("perceiver hierarchy visualization")
    fig.clf()
    axes = fig.add_subplot(1,1,1)
    axes.set_aspect('equal')
    axes.set_xlim(-(text_distance ** 2), text_distance ** 2)
    axes.set_ylim(-(text_distance ** 2), text_distance ** 2)
    axes.hold(True)
    
    """ Draw the lines linking groups with the feauture groups 
    they are composed of.
    """
    """ Find the line weight.
    The weight of each line is the count of the number of features
    from one group that are inputs to the other.
    """
    line_weight = np.zeros((n_nodes - 3, n_nodes), 'float')
    for group_index in range(n_nodes - 3):
        for member_index in range(perceiver.\
                          grouping_map_group.features[group_index].size):
            node_index =  perceiver.grouping_map_group.features[group_index] \
                                          [member_index] + 3
            line_weight[group_index, node_index] += 1
            
    """ Draw the lines between group nodes. 
    The line width is a function of the line weight.
    """
    for group_index in range(n_nodes - 3):
        for node_index in range(n_nodes):
            weight = np.log2(1. +line_weight[group_index, node_index])
            if weight > 0:
                plt.plot((nodes[group_index + 3, 0] , \
                          nodes[node_index     , 0]), \
                         (nodes[group_index + 3, 1] , \
                          nodes[node_index     , 1]), \
                         color='black', linewidth=weight, \
                         solid_capstyle='butt')
                
    """ Draw markers representing each group, and give each
    its appropriate label.
    """ 
    for node_index in range(n_nodes):
        plt.plot(nodes[node_index, 0], nodes[node_index, 1], marker='o',\
                 color='black', markersize=20)
        plt.plot(nodes[node_index, 0], nodes[node_index, 1], marker='o',\
                 color='white', markersize=12)
        plt.plot(nodes[node_index, 0], nodes[node_index, 1], marker='o',\
                 color='black', markersize=6)
        plt.text(nodes[node_index, 0] * text_distance, \
                 nodes[node_index, 1] * text_distance, \
                 node_text[node_index], \
                 horizontalalignment='center', \
                 verticalalignment='center', \
                 size='x-large')        
    
    plt.title("Group hierarchy")
    plt.draw()
    
    if save_eps:
        fig.savefig(epsfilename, format='eps')
        
    return            
        

def visualize_feature_set(grouper, save_eps=False, 
                          epsfilename='log/features.eps'):
    """ Visualize all the groups in all the features """
    label = 'feature_set'
    fig = plt.figure(label)
    fig.clf()
    plt.ioff()
    
    viz_axes = fig.add_subplot(1,1,1)
    plt.title(label)
    
    reduced_features = reduce_feature_set(grouper)  

    pos_ctr = 0.0
    n_feature_groups = len(reduced_features)
    for group_index in range(n_feature_groups):
        current_group = reduced_features[group_index]        
        n_features = len(current_group)
        
        if n_features > 0:
            plt.text(1.0, pos_ctr - 0.15, 'Group' + str(group_index))

        for feature_index in range(n_features):

            visualize_state(current_group[feature_index], 
                            y_max=pos_ctr-0.25,
                            y_min=pos_ctr-0.75,
                            axes=viz_axes)
            pos_ctr -= 1.0
    
    """ This trick makes matplotlib recognize that it has something to plot.
    Everything else in the plot is patches and text, and for some reason
    it doesn't draw unless you include this plot command.
    """ 
    plt.plot(0, 0, color='black') 
    
    force_redraw()
      
    if save_eps:
        fig.savefig(epsfilename, format='eps')
        
    return
  
      
def visualize_feature_spacing(grouper, save_eps=False, 
                          epsfilename='log/features.eps'):
    """ Visualize all the groups in all the features """
    label = 'feature_spacing'
    fig = plt.figure(label)
    fig.clf()
    plt.ioff()
    plt.title(label)
    
    distances = np.zeros((0,1))
    fmap = grouper.feature_map
    n_feature_groups = len(fmap.features)
    for group_index in range(n_feature_groups):
        for feature_index in range(fmap.features[group_index].shape[0]):
            similarities = utils.similarity( 
                fmap.features[group_index][feature_index,:], 
                fmap.features[group_index].transpose())
            
            similarities = np.delete(similarities, [feature_index])
            similarities = similarities[:,np.newaxis]
            distances = np.concatenate((distances, 1-similarities))
            
    if n_feature_groups > 0:
        plt.hist(distances, bins=120)
        force_redraw()
              
        if save_eps:
            fig.savefig(epsfilename, format='eps')
            
    return
  
      
def visualize_feature(grouper, group, feature, label=None):
    """ Visualize a feature or list of features """
    
    if len(group) != len(feature):
        print "error in Visualizer.visualize_feature()"
        print "Length of group and feature lists must be the same."
        print "The group list has ", len(group), \
                " elements and the feature list has ", len(feature)
                
    """ Create a state with the listed features active, 
    then visualize that.
    """
    group_state = perceiver.previous_input.zeros_like()
    
    for feature_index in range(len(feature)):
        if group[feature_index] == -3:
            group_state.sensors[feature[feature_index]] = 1
        elif group[feature_index] == -2:
            group_state.primitives[feature[feature_index]] = 1
        elif group[feature_index] == -1:
            group_state.action[feature[feature_index]] = 1
        else:
            group_state.features[group[feature_index]] \
                              [feature[feature_index]] = 1
                              
    reduced_group_state = reduce_state(group_state, grouper)
    
    if label == None:
        label = "inputs projecting onto group"
        
    visualize_state(reduced_group_state, label)
    force_redraw()
    
    return
  
  
def reduce_feature_set(perceiver, n_primitives, n_actions):
    """ Take in the feature map and express each feature in terms of 
    the lowest level inputs (sensors, primitives, and actions)
    that excite them.
    """
    """ Each row of the feature map represents a single feature's composition.
    Expand each row until it is represented only in terms of low level
    inputs. Start at the highest non-zero column and work from 
    right to left.  
    """
    """ The highest nonzero column will not be more than the number
    of sensors plus the number of features, which includes
    primitives, actions, and created features, modified to account for 
    python's indexing and range behavior.
    """
    first_feature_index = perceiver.n_sensors + perceiver.n_features -1
    
    """ The last feature to be expanded will be the first created 
    feature, i.e. the number of sensors plus 
    the number of primitives and actions, modified to account for 
    python's indexing and range behavior.
    """
    last_feature_index = perceiver.n_sensors + n_primitives + n_actions - 1
    
    reduced_feature_set = copy.deepcopy(
                            perceiver.feature_map[:perceiver.n_features, 
                                                  :first_feature_index + 1])
    
    #print 'first', first_feature_index, 'last', last_feature_index
    
    for column_index in range(first_feature_index, last_feature_index, -1):
        #print 'column_index', column_index

        features_that_have_this_feature_as_input = \
            np.nonzero(reduced_feature_set[:,column_index])[0]
            
        #print 'features_that_have_this_feature_as_input', features_that_have_this_feature_as_input
        #print 'shape', features_that_have_this_feature_as_input.shape
        
        if features_that_have_this_feature_as_input.size > 0:
            
            input_magnitudes = reduced_feature_set[
                       features_that_have_this_feature_as_input, column_index]
            
            #print 'input_magnitudes', input_magnitudes
            
            feature_index = column_index - perceiver.n_sensors
            
            """ Add the contribution from the features in terms of its lower level 
            representation.
            """
            #print 'reduced_feature_set before', reduced_feature_set[features_that_have_this_feature_as_input,:] 
            #print 'shape', reduced_feature_set[features_that_have_this_feature_as_input,:].shape
            #print 'reduced_feature_set[feature_index,:]', reduced_feature_set[feature_index,:]
            #print 'shape', reduced_feature_set[feature_index,:].shape
            
            this_feature = reduced_feature_set[feature_index,:]
            
            #print 'this_feature[:, np.newaxis].shape'
            #print this_feature[:, np.newaxis].shape
            #print 'input_magnitudes[np.newaxis, :].shape'
            #print input_magnitudes[np.newaxis, :].shape

            reduced_feature_set[features_that_have_this_feature_as_input,:] += \
                np.dot(input_magnitudes[:,np.newaxis], this_feature[np.newaxis,:], )
                
            #print 'reduced_feature_set after', reduced_feature_set[features_that_have_this_feature_as_input,:] 
    
            """ Remove the high level representation, just to clean up """
            reduced_feature_set[features_that_have_this_feature_as_input,column_index] = 0
            
            #print 'reduced_feature_set cleanup', reduced_feature_set[features_that_have_this_feature_as_input,:] 
    return reduced_feature_set[n_primitives + n_actions:,:perceiver.n_sensors]

    
def visualize_model(model, n=None):
    """ Visualize some of the transitions in the model """
    if n == None:
        n = model.n_transitions
        
    n = np.minimum(n, model.n_transitions)
        
    print "The model has a total of ", model.n_transitions, \
            " transitions."
    
    '''
    """ Show the n transitions from the model that have the 
    highest count.
    NOTE: argsort returns indices of sort in *ascending* order. 
    """
    index_by_rank = np.argsort(model.count[:model.n_transitions])
    
    for index in range(n):
        print "Showing the " + str(index) + \
                    "th most often observed transition."
        
        visualize_transition(model, index_by_rank[-(index+1)])
        """ Hold the plot, blocking the program until the user closes
        the figure window.
        """
        plt.show()
        
    """ Show the n transitions from the model that have the highest reward """
    index_by_rank = np.argsort(model.reward_value[:model.n_transitions])
    
    for index in range(n):
        print "Showing the " + str(index) + \
                    "th most rewarding transition."
        
        visualize_transition(model, index_by_rank[-(index+1)])
        """ Hold the plot, blocking the program until the user closes
        the figure window.
        """
        plt.show()
    '''    
    """ Show the n transitions from the model that have the 
    highest impact.
    NOTE: argsort returns indices of sort in *ascending* order. 
    """
    index_by_rank = np.argsort(model.reward_value[:model.n_transitions] * \
            (np.log(model.count[:model.n_transitions] + 1)), axis=0).ravel()
    
    for index in range(n):
        print "Showing the " + str(index) + \
                    "th most impact."
                    
        visualize_transition(model, index_by_rank[-(index+1)])
        """ Hold the plot, blocking the program until the user closes
        the figure window.
        """
        plt.show()

    return
    
        
def visualize_transition(model, transition_index, save_eps=False, 
                          label=None, epsfilename='log/transition.eps'):
    """ Visualize a single model transition """
    if label==None:
        label = 'Transition ' + str(transition_index)
        
    n_features = model.n_features
        
    fig = plt.figure(label)
    fig.clf()
    plt.ioff()
    
    viz_axes = fig.add_subplot(1,1,1)
    
    count = model.count[transition_index, 0]
    reward_value = model.reward_value[transition_index, 0]
    reward_uncertainty = model.reward_uncertainty[transition_index, 0]
    goal_value = model.goal_value[transition_index, 0]
    
    plt.title('Transition {:}'.format(transition_index) + 
              '  count: {:.2f}'.format(count)  + 
              '  reward value: {:.2f}'.format(reward_value) + 
              '  uncertainty: {:.2f}'.format(reward_uncertainty) + 
              '  goal value: {:.2f}'.format(goal_value) )
    plt.xlabel(label)
    
    n_primitives = model.context.num_primitives
    n_actions = model.context.num_actions
    
    context = state.State(n_primitives, n_actions, n_features)
    cause = state.State(n_primitives, n_actions, n_features)
    effect = state.State(n_primitives, n_actions, n_features)
    effect_uncertainty = state.State(n_primitives, n_actions, n_features)

    context.features = copy.deepcopy(model.context. \
                        features[:n_features, transition_index, np.newaxis])
    cause.features = copy.deepcopy(model.cause. \
                        features[:n_features, transition_index, np.newaxis])
    effect.features = copy.deepcopy(model.effect. \
                        features[:n_features, transition_index, np.newaxis])
    effect_uncertainty.features = copy.deepcopy(model.effect_uncertainty. \
                        features[:n_features, transition_index, np.newaxis])

    visualize_state(context, y_max=3.75, y_min=3.25, axes=viz_axes)
    visualize_state(cause, y_max=2.75, y_min=2.25, axes=viz_axes)
    visualize_state(effect, y_max=1.75, y_min=1.25, axes=viz_axes)
    visualize_state(effect_uncertainty, y_max=0.75, y_min=0.25, axes=viz_axes)
    
    """ This trick makes matplotlib recognize that it has something to plot.
    Everything else in the plot is patches and text, and for some reason
    it doesn't draw unless you include this plot command.
    """ 
    plt.plot(0, 0, color='black') 
    
    force_redraw()
      
    if save_eps:
        fig.savefig(epsfilename, format='eps')
        
    return
  
  
def visualize_state(state, label='state', y_min=0.25, y_max=0.75, 
                      save_eps=False, epsfilename='log/state.eps', 
                      axes=None):
    """ Present the state in a visually intuitive way.
    height_proportion is the height of the display as 
    a fraction of width.
    """       
    if axes == None:
        fig = plt.figure(label)
        fig.clf()
        axes = fig.add_subplot(1,1,1)
        # TODO: make background color light gray
        plt.title(label)
        axes.set_ylim(0.0, 1.0)
     
    """ The number of blank features in between groups """
    x_spacer_width = 3 
    total_width = 0

    primitives = state.get_primitives()
    actions = state.get_actions()
    features = state.features[primitives.size + actions.size:]
    
    #print 'primitives', primitives.ravel()
    #print 'actions', actions.ravel()
    #print 'features', features.ravel()
    
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
    
    """ Get the figure to recognize that it has something to plot.
    A blatant hack.  """       
    plt.plot(0, 0, color='black') 

    if save_eps:
        fig.savefig(epsfilename, format='eps')
        
    return


def visualize_array_list(array_list, label=None):
    """ Show a list of arrays as a set of line plots in a single figure.
    Useful for tracking the time history of a set of values.
    """
    if len(array_list) == 0:
        return
    
    if label == None:
        label = 'arrays'
        
    """ Condense all the arrays into a single 2D array """
    n_cols = len(array_list)
    n_rows = array_list[0].size
    master_array = np.zeros((n_rows, n_cols))
    
    for i in range(n_cols):
        master_array[:,i] = array_list[i].ravel()
    
    plt.figure(label)
    plt.clf()
    plt.hold(True)
    for i in range(n_rows):
        plt.plot(master_array[i,:])
    
    plt.title(label)
    plt.draw()


def rectPatch (left_x, right_x, lower_y, upper_y, value=0.5, 
           axes=plt.gca(), borders=False):
    """ Draws a rectangular patch bounded by (left_x, lower_y) on
    the lower, left hand corner and (right_x, upper_y) on the
    upper, right hand corner. The grayscale value of the patch can 
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
    
    
def force_redraw():
    """
    Force matplotlib to draw things on the screen
    """
    
    """ Pause is needed for events to be processed
    Qt backend needs two event rounds to process screen. 
    Any number > 0.01 and <=0.02 would do
    """
    plt.pause(0.015)

