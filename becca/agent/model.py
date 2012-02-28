
import logging
import copy

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from ..utils import bounded_sum, similarity, force_redraw

class Model(object):

    def __init__(self, num_primitives, num_actions, graphs=True):

        self.create_logger()
        
        self.SIMILARITY_THRESHOLD = 0.9       # real, 0 < x < 1
        self.MAX_ENTRIES = 10 ** 4               # integer, somewhat large
        self.CLEANING_PERIOD = 10 ** 5           # integer, somewhat large
        self.UPDATE_RATE = 0.1                # real, 0 < x < 1
        self.STEP_DISCOUNT = 0.5              # real, 0 < x < 1
        self.TRACE_LENGTH = 10                 # integer, small
        self.TRACE_DECAY_RATE = 0.2           # real, 0 < x < 1

        self.clean_count = 0

        self.cause = [[]]
        
        self.cause.append(np.zeros((num_primitives, 2*self.MAX_ENTRIES)))
        self.cause.append(np.zeros((num_actions, 2*self.MAX_ENTRIES)))

        self.effect = copy.deepcopy(self.cause)
        self.history = copy.deepcopy(self.cause)

        self.count = np.zeros(2*self.MAX_ENTRIES)
        self.reward_map = np.zeros(2*self.MAX_ENTRIES)
        self.goal_map = np.zeros(2*self.MAX_ENTRIES)

        self.trace_index = np.ones(self.TRACE_LENGTH)
        self.trace_reward = np.zeros(self.TRACE_LENGTH)

        # Initializes dummy transitions
        self.last_entry = 1
        self.cause[1][0,0] = 1
        self.effect[1][0,0] = 1
        #self.countp[0]= eps

        self.graphing = graphs


    def create_logger(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_group(self):
        size = (1, np.size(self.cause[-1], 1))
        self.history.append(np.zeros(size))
        self.cause.append(np.zeros(size))
        self.effect.append(np.zeros(size))


    def add_feature(self, nth_group, has_dummy):
        """
        add_feature to the model
        """
        
        self.history[nth_group] = np.vstack((self.history[nth_group],  np.zeros(self.history[nth_group].shape[1])))
        self.cause[nth_group]   = np.vstack((self.cause[nth_group],  np.zeros(self.cause[nth_group].shape[1])))
        self.effect[nth_group]  = np.vstack((self.effect[nth_group], np.zeros(self.effect[nth_group].shape[1])))
        
        # if dummy feature is still in place, removes it
        if has_dummy:
            self.history[nth_group]   = self.history[nth_group][1:]
            self.cause[nth_group]  = self.cause[nth_group][1:]
            self.effect[nth_group] = self.effect[nth_group][1:]



    def train(self, new_effect, new_history, new_cause, reward):
        """
        Takes in 'new_cause' and 'new_effect' 
        to train the model of the environment.
        """

        # TODO: udpate comments to reflect new representation

        eps = np.finfo(np.double).eps            
        num_groups = len(new_cause)

        # Checks to see whether the new entry is already in the library 
        # TODO: make the similarity threshold a function of the count? This would
        # allow often-observed transitions to require a closer fit, and populate
        # the model space more densely in areas where it accumulates more
        # observations.
        transition_similarity = similarity(new_history, self.history, range(self.last_entry))

        # If the cause doesn't match, the transition doesn't match
        cause_group = None
        cause_feature = None
        for group_index in range(1, num_groups):
            group = new_cause[group_index]
            feature = group.ravel().nonzero()[0]
            if np.size(feature):
                cause_group = group_index
                cause_feature = feature
                 
        match_indices = []
        
        if cause_group is not None:            
            transition_similarity *= self.cause[cause_group][cause_feature, :self.last_entry][0]
            match_indices = ( transition_similarity > self.SIMILARITY_THRESHOLD).ravel().nonzero()[0]


        #if context and cause are sufficiently different, 
        #adds a new entry to the model library
        if np.size(match_indices) == 0: 
            self.last_entry += 1
            matching_transition_index = self.last_entry
            for index in range(1,num_groups):
                self.history[index][:, self.last_entry] = new_history[index]
                self.cause[index][:, self.last_entry] = new_cause[index]
                self.effect[index][:, self.last_entry] = new_effect[index]

            self.count[matching_transition_index] =  1
            current_update_rate = 1.0

        #otherwise increments a nearby entry
        else:
            matching_transition_index = np.argmax(transition_similarity)

            self.count[matching_transition_index] += 1

            # modifies cause and effect
            # making the update rate a function of count allows updates to occur
            # more rapidly when there is little past experience to contradict them
            current_update_rate = (1 - self.UPDATE_RATE) / self.count[matching_transition_index] + self.UPDATE_RATE
            current_update_rate = min(1.0, current_update_rate)

            for index in range(1,num_groups):
                self.effect[index][:, matching_transition_index] = self.effect[index][:, matching_transition_index] * \
                    (1 - current_update_rate) + new_effect[index] * current_update_rate
                # TODO: update cause as well?
                #         self.cause[k][:, matching_transition_index] = 

                
        #Performs credit assignment on the trace
        # updates the transition trace
        self.trace_index = np.hstack((self.trace_index, matching_transition_index))
        self.trace_index = self.trace_index[1:]

        # updates the reward trace
        self.trace_reward = np.hstack((self.trace_reward, reward))
        self.trace_reward = self.trace_reward[1:]

        credit = self.trace_reward[0]
        for i in range(1, self.TRACE_LENGTH):
            credit = bounded_sum( credit, self.trace_reward[i] * (1 - self.TRACE_DECAY_RATE) ** i)


        # updates the reward associated with the last entry in the trace
        update_index = self.trace_index[0]
        max_step_size = credit - self.reward_map[update_index]

        # # without assigning credit to the trace
        # update_index = matching_transition_index
        # max_step_size = reward - self.reward_map[update_index]

        self.reward_map[update_index] += max_step_size * current_update_rate


        #########################        
        self.clean_count += 1

        # cleans out the library
        if self.last_entry >= self.MAX_ENTRIES:
            self.clean_count = self.CLEANING_PERIOD + 1

        if self.clean_count > self.CLEANING_PERIOD:
            self.logger.info("Cleaning up model")
            
            self.count[:self.last_entry] -=  1 / self.count[:self.last_entry]
            forget_indices = (self.count[:self.last_entry] < eps).ravel().nonzero()[0]

            for index in range(1,num_groups):
                self.history[index] = np.delete(self.history[index], forget_indices, 1)
                self.cause[index] = np.delete(self.cause[index], forget_indices, 1)
                self.effect[index] = np.delete(self.effect[index], forget_indices, 1)

            self.count = np.delete(self.count, forget_indices, 1)
            self.reward_map = np.delete(self.reward_map, forget_indices, 1)

            self.clean_count = 0
            self.last_entry -= len(forget_indices)
            if self.last_entry < 0:
                self.last_entry = 0

            #debug
            self.logger.debug('Library cleaning out %s entries to %s entries' % (len(forget_indices), self.last_entry))



        # pads the library if it has shrunk too far
        if np.size(self.effect[1], 1) < self.MAX_ENTRIES * 1.1:
            
            for index in range(1,num_groups):
                size = (self.effect[index].shape[0], self.MAX_ENTRIES)
                self.history[index] = np.hstack((self.history[index], np.zeros(size)))
                self.cause[index]  = np.hstack((self.cause[index], np.zeros(size)))
                self.effect[index] = np.hstack((self.effect[index], np.zeros(size)))

  
            self.count = np.hstack((self.count, np.zeros(self.MAX_ENTRIES)))
            self.reward_map = np.hstack((self.reward_map, np.zeros(self.MAX_ENTRIES)))
  

            
    def display_n_best(self, N):
        """
        provides a visual representation of the Nth cause-effect pair
        """


        sorted_indices = np.argsort(self.count[:self.last_entry+1])[::-1]
        relevant_sorted_indices = sorted_indices[:N]

        print relevant_sorted_indices
        for order, index in enumerate(relevant_sorted_indices):
            self.display_pair(index, "%sth top causes and effects" % (order + 1))


    def display_pair(self, N, figure_name):
        """
        provides a visual representation of the Nth cause-effect pair
        """

        import time

        if self.graphing:
            # TODO: this can be sped up by keeping the axes returned from subplots and drawing directly there
            # see: http://stackoverflow.com/questions/8798040/optimizing-matplotlib-pyplot-plotting-for-many-small-plots
            
            
            start = time.time()
            plt.figure(figure_name)
            plt.clf()  # if we don't clear, the bars overlap. 
            
            for index in range(1, num_groups):
                plt.subplot(num_groups-1, 2, index*2-1)
                heights = self.cause[index][:,N]
                plt.bar(np.arange(len(heights)), heights)
                #self.logger.info('model.cause[%s][%s]' % (index, N))
                #self.logger.info(self.cause[index][:,N]
                plt.axis([0, self.cause[index].shape[0]+1, 0, 1])
                plt.ylabel("Group %s" % index)
                #    xlabel(['max of ' num2str(max(self.cause[index](:,N))) ])
                if index == 1:
                    plt.title('model.cause for N = %s' % N)

                if index == num_groups - 1:
                    plt.xlabel('count = %s' % self.count[index])


            for index in range(1, num_groups):
                plt.subplot(num_groups-1, 2, index*2)
                heights = self.effect[index][:,N]
                plt.bar(np.arange(len(heights)), heights)
                #self.logger.info('model.effect[%s][%s]' % (index, N))
                #self.logger.info(self.effect[index][:,N])
                plt.axis ([0, self.effect[index].shape[0]+1, 0, 1])
                plt.ylabel("Group %s" % index)
                plt.xlabel('max of %s' % np.max(self.effect[index][:,N]))
                if index == 1:
                    plt.title('model.effect for N = %s' % N)

            force_redraw() # if no redraw is forced, only the first pair created will be redrawn


            
    def display(self, N):
        """
        provides a visual representation of the Nth cause-effect pair
        """

        if self.graphing:
            num_groups = len(self.cause)
            # 
            plt.figure("causes and effects")
            for index in range(1, num_groups):
                plt.subplot(num_groups - 1, 2, (index-1)*2-1)
                plt.bar(np.arange(len(self.cause[index][:,N])), self.cause[index][:,N])
                plt.axis (0, self.cause[index].shape[0] + 1, 0, 1)
                plt.ylabel('group %s' % index)
                if index == 1:
                    plt.title('cause for N = %s' % N)

                if index == num_groups - 1:
                    plt.xlabel('count = %s' % self.count[N])
            # 
            # for index = 2:num_groups,
            #     subplot(num_groups-1, 2, (index-1)*2)
            #     bar(self.effect[index](:,N))
            #     axis ([0 size(self.effect[index],1)+1 0 1])
            #     ylabel(['grp ' num2str(index) ])
            #     if (index == 2)
            #         title(['self.effect for N = ' num2str(N) ])
            #     end
            # end

        # Text display
        self.logger.info('================Transition pair %s =================' % N)
        for index in range(1, num_groups):
            nonz_cause = self.cause[index][:,N].ravel().count_nonzero()
            nonz_effect = self.effect[index][:,N].ravel().count_nonzero()
            if (nonz_cause > 0) or (nonz_effect > 0):

            #     # debug
            #     if ( (nonz_cause < 4) && (nonz_effect < 4))
            #         

                self.logger.info('  -----Group %s -----' % index)
                for subindex in range(self.effect[index][:,N].shape[0]):
                    if self.cause[index][subindex,N].count_nonzero() or self.effect[index][subindex,N].count_nonzero():
                        self.logger.info('    Feature             %s: %s %s' % (subindex, self.cause[index][subindex,N],
                                                                                self.effect[index][subindex,N]))


    




     
