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

def visualize_grouper_coactivity(coactivity, size=0, 
                                    save_eps=False, 
                                    epsfilename='log/coactivity.eps'):
    """ Produce a visual representation of the coactivity matrix """
    
    if size == 0:
        size = coactivity.shape[0]
    fig = plt.figure("perceiver coactivity visualization")
    # Diane L. made the brilliant suggestion to leave this plot in color. 
    # It looks much prettier.
    plt.summer()
    #plt.hot()
    im = plt.imshow(coactivity[0:size, 0:size])
    im.set_interpolation('nearest')
    plt.title("Coactivity among inputs")
    plt.draw()
    
    if save_eps:
        fig.savefig(epsfilename, format='eps')
        
    return
  
    
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
  
      
def reduce_feature_set(perceiver):
    """ Reduce the entire feature set (every feature from every group) 
    to their low-level constituents in terms of sensors, primitives, 
    and action.
    Returns a list of lists of State objects.
    """
    
    n_feature_groups = perceiver.previous_input.n_feature_groups()    
    reduced_features = []

    for group_index in range(n_feature_groups):
        current_group = perceiver.previous_input.features[group_index]
        n_features = current_group.size
        reduced_features_this_group = []
    
        for feature_index in range(n_features):
            current_feature_state = perceiver.previous_input.zeros_like()            
            current_feature_state.features[group_index][feature_index] = 1.0           
            reduced_state = reduce_state(current_feature_state, perceiver)
            reduced_features_this_group.append(reduced_state)   
        
        reduced_features.append(reduced_features_this_group)
                            
    return reduced_features
        
        
def visualize_model(model, n=None):
    """ Visualize some of the transitions in the model """
    if n == None:
        n = model.n_inputs
        
    n = np.minimum(n, model.n_inputs)
        
    print "The model has a total of ", model.n_inputs, \
            " transitions."
    
    '''
    """ Show the n transitions from the model that have the 
    highest count.
    NOTE: argsort returns indices of sort in *ascending* order. 
    """
    index_by_rank = np.argsort(model.count[:model.n_inputs])
    
    for index in range(n):
        print "Showing the " + str(index) + \
                    "th most often observed transition."
        
        visualize_transition(model, index_by_rank[-(index+1)])
        """ Hold the plot, blocking the program until the user closes
        the figure window.
        """
        plt.show()
        
    """ Show the n transitions from the model that have the highest reward """
    index_by_rank = np.argsort(model.reward_value[:model.n_inputs])
    
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
    index_by_rank = np.argsort(model.count[:model.n_inputs] * \
                        (np.log(model.reward_value[:model.n_inputs] + \
                                np.ones(model.n_inputs))))
    
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
    
    context = state.State()
    cause = state.State()
    effect = state.State()
    effect_uncertainty = state.State()
    
    context.sensors = np.zeros((0,0))
    cause.sensors = np.zeros((0,0))
    effect.sensors = np.zeros((0,0))
    effect_uncertainty.sensors = np.zeros((0,0))
    
    context.primitives = model.context.primitives[:,transition_index]
    cause.primitives = model.cause.primitives[:,transition_index]
    effect.primitives = model.effect.primitives[:,transition_index]
    effect_uncertainty.primitives = \
                    model.effect_uncertainty.primitives[:,transition_index]
    
    context.action = model.context.action[:,transition_index]
    cause.action = model.cause.action[:,transition_index]
    effect.action = model.effect.action[:,transition_index]
    effect_uncertainty.action = model.effect_uncertainty.action[:,transition_index]
    
    for group_index in range(len(model.context.features)):
        context.features.append(model.context.features \
                                [group_index][:,transition_index])
        cause.features.append(model.cause.features \
                              [group_index][:,transition_index])
        effect.features.append(model.effect.features \
                               [group_index][:,transition_index])
        effect_uncertainty.features.append(model.effect_uncertainty.features \
                               [group_index][:,transition_index])
        
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
  
                    
def reduce_state(full_state, perceiver):
    """ Reduce a state, projecting it down to a representation in only
    sensors, primitives, and action. 
    Returns a state, the same size as the input state, in which 
    only the sensors, primitives, and action have non-zero elements.
    """
    
    """ Expand any active features, group by group, starting at the
    highest-numbered group and counting down. 
    Features from higher numbered groups always reduce into lower numbered
    groups. Counting down allows for a cascading expansion of high-level
    features into their lower level component parts, until they are
    completely broken down into sensors, primitives, and action.
    
    Lower numbered feature groups that contribute features as inputs
    to higher numbered feature groups are considered their parents.
    The sensors, primitives, and action groups are the original parents 
    of all other features.
    """
    state = copy.deepcopy(full_state)
    for group_index in range(state.n_feature_groups() - 1, -1, -1):
        """ Check whether there are any nonzero elements in 
        each group that need to be expanded.
        """
        if np.count_nonzero(state.features[group_index]):
            """ Find contributions of all lower groups to the 
            current group, again counting down to allow for 
            cascading downward projections. parent_group_index 
            is a group counter variable, specific to propogating 
            downward projections. 
            """
            for parent_group_index in range(group_index-1, -4, -1):
                match_indices = \
                        (perceiver.grouping_map_group.features[group_index] == 
                         parent_group_index).nonzero()[0]
                """print 'match_indices for group ', group_index, \
                        ' in group ', parent_group_index, ' ', \
                         match_indices
                """
                parent_feature_indices = \
                         perceiver.grouping_map_feature.features[group_index] \
                         [match_indices,:]
                """print 'parent_feature_indices for group ', group_index, \
                        ' in group ', parent_group_index, ' ', \
                         parent_feature_indices
                """
    
                """ Check whether the parent group is actually a
                parent of the group being reduced.
                """
                if parent_feature_indices.size:
    
                    """ Expand each relevant feature from the
                    group being reduced down to the lower level features 
                    in each parent group.
                    """
                    for feature_index in range \
                            (state.features[group_index].size):
    
                        """ Check whether the feature to reduce  
                        is nonzero. This check saves a lot of 
                        unnecessary computation.
                        """
                        this_feature_activity = \
                                state.features[group_index][feature_index]
                        if this_feature_activity != 0:
                                    
                            """ Translate the feature activity to its 
                            lower level parent features.
                            propagation_strength is the amount that
                            each input to the group contributes
                            to the feature being reduced. 
                            """
                            """ Use this variant to propogate straight 
                            feature representations. It makes accurate,
                            but sometimes uninteresting feature displays,
                            because feature representations tend to cluster
                            toward the middle of the group space.
                            """
                            #
                            #propagation_strength = perceiver.feature_map.\
                            #       features[group_index] \
                            #       [feature_index, match_indices]
                            
                            """ This variant, using get_perceptive_field,
                            emphasizes the differences between features.
                            It makes for prettier and more informative
                            displays of feature sets.
                            """
                            propagation_strength = perceiver.feature_map. \
                                    get_receptive_field(group_index, 
                                                    feature_index) \
                                                    [match_indices]
    
                            """ propagated_activation is the propagation
                            strength scaled by the activity of the
                            feature being reduced.
                            """
                            propagated_activation = this_feature_activity * \
                                propagation_strength.transpose()[:,np.newaxis]
                                    
    
                            """ The lower-level feature is incremented 
                            according to the propagated_activation. The
                            incrementing is done nonlinearly to ensure
                            that the activity of the lower level features
                            never exceeds 1.
                            """
                            """ Handle sensors, primitives, and action
                            separately.
                            """
                            if parent_group_index == -3:
                                state.sensors \
                                        [parent_feature_indices.ravel()] = \
                                        utils.bounded_sum( \
                                        state.sensors \
                                        [parent_feature_indices.ravel()], \
                                        propagated_activation)
                            elif parent_group_index == -2:
                                state.primitives \
                                        [parent_feature_indices.ravel()] = \
                                        utils.bounded_sum( \
                                        state.primitives \
                                        [parent_feature_indices.ravel()], \
                                        propagated_activation)
                            elif parent_group_index == -1:
                                state.action \
                                        [parent_feature_indices.ravel()] = \
                                        utils.bounded_sum( \
                                        state.action \
                                        [parent_feature_indices.ravel()], \
                                        propagated_activation)
                            else:
                                state.features[parent_group_index] \
                                        [parent_feature_indices.ravel()] = \
                                        utils.bounded_sum( \
                                        state.features[parent_group_index] \
                                        [parent_feature_indices.ravel()], \
                                        propagated_activation)
    
            """ Eliminate the original representation of 
            the reduced feature, now that it is expressed in terms 
            of its lower level parent features.
            """
            #state.features[group_index] = \
            #        np.zeros(state.features[group_index].shape)
    state.features = []
    return state


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

    x_spacer_width = 3 # number of blank features in between groups

    """ Handle the case when the sensors group is None """
    if state.sensors == None:
        state.sensors = np.zeros((0,0))
        
    total_width = state.sensors.size
    total_width += x_spacer_width
    total_width += state.primitives.size
    total_width += x_spacer_width
    total_width += state.action.size
    
    for indx in range(state.n_feature_groups()):
        total_width += x_spacer_width
        total_width += state.features[indx].size
            
    axes.set_xlim(-x_spacer_width, total_width + x_spacer_width)
        
    x = 0
    for indx in range(state.sensors.size):
        rectPatch(x, x + 1, y_min, y_max, state.sensors[indx], axes)
        x += 1
        
    x += x_spacer_width
    for indx in range(state.primitives.size):
        rectPatch(x, x + 1, y_min, y_max, state.primitives[indx], axes)
        x += 1
        
    x += x_spacer_width
    for indx in range(state.action.size):
        rectPatch(x, x + 1, y_min, y_max, state.action[indx], axes)
        x += 1
    
    for feature_group_indx in range(state.n_feature_groups()):    
        x += x_spacer_width
        for indx in range(state.features[feature_group_indx].size):
            rectPatch(x, x + 1, y_min, y_max,
                           state.features[feature_group_indx][indx], axes)
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

