
import agent.utils as utils
import agent.viz_utils as viz_utils

import matplotlib.pyplot as plt
import numpy as np

def center_surround(fov, fov_span, block_heigth, block_width, verbose=False):
    super_pixels = np.zeros((fov_span + 2, fov_span + 2))
    center_surround_pixels = np.zeros((fov_span, fov_span))
    
    """ center-surround pixels """
    for row in range(fov_span + 2):
        for column in range(fov_span + 2):
            super_pixels[row][column] = np.mean( fov[row * block_heigth: (row + 1) * block_heigth , 
                                                     column * block_width: (column + 1) * block_width ])                
    for row in range(fov_span):
        for column in range(fov_span):
            
            """ Calculate a center-surround value that represents
            the difference between the pixel and its surroundings.
            """
            center_surround_pixels[row][column] = \
                super_pixels[row + 1][column + 1] - \
                super_pixels[row    ][column + 1] / 6 - \
                super_pixels[row + 2][column + 1] / 6 - \
                super_pixels[row + 1][column    ] / 6 - \
                super_pixels[row + 1][column + 2] / 6 - \
                super_pixels[row    ][column    ] / 12 - \
                super_pixels[row + 2][column    ] / 12 - \
                super_pixels[row    ][column + 2] / 12 - \
                super_pixels[row + 2][column + 2] / 12
                
    if verbose:
        plt.figure("fov")
        plt.gray()
        im = plt.imshow(fov)
        im.set_interpolation('nearest')
        plt.title("field of view")
        plt.draw() 
        
        plt.figure("super_pixels")
        plt.gray()
        im = plt.imshow(super_pixels)
        im.set_interpolation('nearest')
        plt.title("super pixels")
        plt.draw() 
        
        plt.figure("center_surround_pixels")
        plt.gray()
        im = plt.imshow(center_surround_pixels)
        im.set_interpolation('nearest')
        plt.title("center surround pixels")
        plt.draw() 
        
        viz_utils.force_redraw()
        
    return center_surround_pixels


def vizualize_pixel_array_feature(feature, level_index, feature_index, world_name=None, save_eps=False, save_jpg=False,
                                  filename='log/feature'):
    """ Calculate the number of pixels that span the field of view """
    n_pixels = feature.shape[0]/ 2
    fov_span = np.sqrt(n_pixels)
    
    level_str = str(level_index).zfill(2)
    feature_str = str(feature_index).zfill(3)
    fig_title = 'Level ' + level_str + ' Feature ' + feature_str + ' from ' + world_name
    fig_name = 'Features from ' + world_name
    fig = plt.figure(utils.ord_str(fig_name))
    fig.clf()
    num_states = feature.shape[1]
    for state_index in range(num_states):
        feature_sensors = feature[:,state_index]
        """ Maximize contrast """
        feature_sensors *= 1 / (np.max(feature_sensors) + 10 ** -6)
        pixel_values = ((feature_sensors[ 0:n_pixels] - feature_sensors[n_pixels:2 * n_pixels]) + 1.0) / 2.0
        feature_pixels = pixel_values.reshape(fov_span, fov_span)
        plt.gray()
        ax = fig.add_axes((float(state_index)/float(num_states), 0., 1/float(num_states), 1.), frame_on=False)
        im = plt.imshow(feature_pixels, vmin=0.0, vmax=1.0, interpolation='nearest')
    if save_eps:
        epsfilename = filename + '_' + world_name  + '_' + level_str + '_' + feature_str + '.eps'
        fig.savefig(epsfilename, format='eps')
    if save_jpg:
        try:
            jpgfilename = filename + '_' + world_name  + '_' + level_str + '_' + feature_str + '.jpg'
            fig.savefig(jpgfilename, format='jpg')
        except:
            print("I think you need to have PIL installed to print in .jpg format.")
            save_jpg = False
    fig.show()
    fig.canvas.draw()
    return
    
