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
    """ Provide an intuitive display of the features created by the 
    agent. 
    """
    """ feature_set is a list of lists of State objects """
    if len(feature_set) == 0:
        return

    """ Calculate the number of pixels that span the field of view """
    n_pixels = feature_set[0][0].sensors.size / 2
    fov_span = np.sqrt(n_pixels)

    """ The number of black pixels surrounding each feature """
    border = 1
    
    """ The number of gray pixels between all features """
    gap = 3
    
    n_groups = len(feature_set)
    '''
    """ Find the size of the overall image_data """
    n_features_max = 0
    for group_index in range(n_groups):
        if len(feature_set[group_index]) > n_features_max:
            n_features_max = len(feature_set[group_index])

    n_pixel_columns = n_features_max * (gap + 2 * border + fov_span) + gap
    n_pixel_rows = n_groups * (gap + 2 * border + fov_span) + gap
    feature_image = 0.8 * np.ones((n_pixel_rows, n_pixel_columns))
    '''
    """ Populate each feature in the feature image_data """
    for group_index in range(n_groups):
        
        """ Find the size of the image for this group """
        n_features = len(feature_set[group_index])
        n_pixel_columns = n_features * (gap + 2 * border + fov_span) + gap
        n_pixel_rows = (gap + 2 * border + fov_span) + gap
        feature_image = 0.8 * np.ones((n_pixel_rows, n_pixel_columns))
    
        for feature_index in range(len(feature_set[group_index])):
            sensors = feature_set[group_index][feature_index].sensors
            
            pixel_values = ((sensors[0:n_pixels] - \
                             sensors[n_pixels:2 * n_pixels]) \
                             + 1.0) / 2.0
            
            feature_pixels = pixel_values.reshape(fov_span, fov_span)
            feature_image_first_row = gap + border 
            feature_image_last_row = feature_image_first_row + fov_span
            
            feature_image_first_column = feature_index * \
                        (gap + 2 * border + fov_span) + gap + border 
            feature_image_last_column = feature_image_first_column + \
                                        fov_span
            feature_image[feature_image_first_row:
                          feature_image_last_row,
                          feature_image_first_column:
                          feature_image_last_column] = feature_pixels
                          
            """ Write North border """
            feature_image[feature_image_first_row - border:
                          feature_image_first_row,
                          feature_image_first_column - border:
                          feature_image_last_column + border] = 0
            """ Write South border """
            feature_image[feature_image_last_row:
                          feature_image_last_row + border,
                          feature_image_first_column - border:
                          feature_image_last_column + border] = 0
            """ Write East border """
            feature_image[feature_image_first_row - border:
                          feature_image_last_row + border,
                          feature_image_first_column - border:
                          feature_image_first_column] = 0
            """ Write West border """
            feature_image[feature_image_first_row - border:
                              feature_image_last_row + border,
                              feature_image_last_column:
                              feature_image_last_column + border] = 0
                
        """ Pad the group number with leading zeros out to three digits """
        group_str = str(group_index).zfill(3)
        fig = plt.figure(world_name + " world features, group " + 
                          group_str)
        plt.gray()
        img = plt.imshow(feature_image, vmin=0.0, vmax=1.0)
        img.set_interpolation('nearest')
        plt.title("Features created while in the " + world_name + 
                  ", group " + group_str)

        """ Save each group's features separately in its own image """
        if save_eps:
            epsfilename = filename + '_' + world_name  + '_' + \
                    group_str + '.eps'
            fig.savefig(epsfilename, format='eps')

        if save_jpg:
            try:
                jpgfilename = filename + '_' + world_name  + '_' + \
                        group_str + '.jpg'
                fig.savefig(jpgfilename, format='jpg')
            except:
                print("I think you need to have PIL installed to print in .jpg format.")
                
        if not(show_image):
            plt.close()
        
    return