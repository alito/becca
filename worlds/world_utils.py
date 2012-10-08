import agent.utils as utils
import matplotlib.pyplot as plt
import numpy as np

""" preprocessing utilities """ 

def center_surround(fov, fov_span, block_heigth, block_width):
    super_pixels = np.zeros((fov_span + 2, fov_span + 2))
    center_surround_pixels = np.zeros((fov_span, fov_span))
    
    """ center-surround pixels """
    for row in range(fov_span + 2):
        for column in range(fov_span + 2):

            super_pixels[row][column] = \
                np.mean( fov[row * block_heigth: (row + 1) * \
                             block_heigth , 
                             column * block_width: (column + 1) * \
                             block_width ])
                
    for row in range(fov_span):
        for column in range(fov_span):
            
            """ Calculate a center-surround value that represents
            the difference between the pixel and its surroundings.
            The result lies between -1 and 1.
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
                
            """ Normalize to scale up small values and to ensure that 
            it falls between 0 and 1.
            """
            center_surround_pixels[row][column] *= 10
            center_surround_pixels[row][column] = \
                utils.map_inf_to_one(center_surround_pixels[row][column])
            center_surround_pixels[row][column] = (1 + \
                center_surround_pixels[row][column]) / 2

    return center_surround_pixels


def vizualize_pixel_array_feature_set(feature_set, world_name=None,
                                  show_image=False,
                                  save_eps=False, save_jpg=False,
                                  filename='log/feature_set'):
    if feature_set.size == 0:
        return

    """ Calculate the number of pixels that span the field of view """
    n_pixels = feature_set.shape[1]/ 2
    fov_span = np.sqrt(n_pixels)
    
    for feature_index in range(feature_set.shape[0]):
        
        feature_sensors = feature_set[feature_index, 0:2 * n_pixels]
 
        """ Maximize contrast """
        feature_sensors *= 1 / (np.max(feature_sensors) + 10**-6)
        
        pixel_values = ((feature_sensors[ 0:n_pixels] - \
                         feature_sensors[n_pixels:2 * n_pixels]) \
                         + 1.0) / 2.0
                         
        feature_pixels = pixel_values.reshape(fov_span, fov_span)
                        
        """ Pad the group number with leading zeros out to three digits """
        feature_str = str(feature_index).zfill(3)
        fig = plt.figure(world_name + " world features, feature " + 
                          feature_str)
        plt.gray()
        img = plt.imshow(feature_pixels, vmin=0.0, vmax=1.0)
        img.set_interpolation('nearest')
        plt.title("Feature " + feature_str + " from " + world_name)
    
        """ Save each group's features separately in its own image """
        if save_eps:
            epsfilename = filename + '_' + world_name  + '_' + \
                    feature_str + '.eps'
            fig.savefig(epsfilename, format='eps')
    
        if save_jpg:
            try:
                jpgfilename = filename + '_' + world_name  + '_' + \
                        feature_str + '.jpg'
                fig.savefig(jpgfilename, format='jpg')
            except:
                print("I think you need to have PIL installed to print in .jpg format.")
                
        if not(show_image):
            plt.close()
   
    return
    