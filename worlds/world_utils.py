import agent.utils as utils
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
