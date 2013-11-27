import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import core.tools as tools

"""
Utilities shared between several worlds dealing with visual input
"""

def center_surround(fov, fov_horz_span, fov_vert_span, verbose=False):
    """ 
    Convert a 2D array of b/w pixel values to center-surround 
    
    fov (field of view) is the 2D array of pixel values and 
    fov_span is the number of center-surround superpixel rows and columns.
    Returns a 2D array of the center surround vales.
    """ 
    fov_height = fov.shape[0]
    fov_width = fov.shape[1]
    block_width = np.round(fov_width / (fov_horz_span + 2))
    block_height = np.round(fov_height / (fov_vert_span + 2))
    super_pixels = np.zeros((fov_vert_span + 2, fov_horz_span + 2))
    center_surround_pixels = np.zeros((fov_vert_span, fov_horz_span))
    # Create the superpixels by averaging pixel blocks
    for row in range(fov_vert_span + 2):
        for column in range(fov_horz_span + 2):
            super_pixels[row][column] = np.mean(
                    fov[row * block_height:(row + 1) * block_height,
                        column * block_width: (column + 1) * block_width ])
    for row in range(fov_vert_span):
        for column in range(fov_horz_span):
            # Calculate a center-surround value that represents
            # the difference between the pixel and its surroundings.
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
        # Display the field of view clipped from the original image
        plt.figure("fov")
        plt.gray()
        im = plt.imshow(fov)
        im.set_interpolation('nearest')
        plt.title("field of view")
        plt.draw() 
        # Display the pixelized version, a.k.a. superpixels
        plt.figure("super_pixels")
        plt.gray()
        im = plt.imshow(super_pixels)
        im.set_interpolation('nearest')
        plt.title("super pixels")
        plt.draw() 
        # Display the center-surround filtered superpixels
        plt.figure("center_surround_pixels")
        plt.gray()
        im = plt.imshow(center_surround_pixels)
        im.set_interpolation('nearest')
        plt.title("center surround pixels")
        plt.draw() 
    return center_surround_pixels

def visualize_pixel_array_feature(feature, 
                                 fov_horz_span=None, fov_vert_span=None,
                                  block_index=-1, feature_index=-1, 
                                  world_name=None, save_png=False, 
                                  filename='log/feature', array_only=False):
    """ Show a visual approximation of an array of center-surround features """
    # Calculate the number of pixels that span the field of view
    n_pixels = feature.shape[0]/ 2
    if fov_horz_span is None:
        fov_horz_span = np.sqrt(n_pixels)
        fov_vert_span = np.sqrt(n_pixels)
    if array_only:
        pixel_array = []
    else:
        block_str = str(block_index).zfill(2)
        feature_str = str(feature_index).zfill(3)
        fig_title = ' '.join(('Block', block_str, 'Feature', feature_str, 
                              'from', world_name))
        fig_name = ' '.join(('Features from ', world_name))
        fig = plt.figure(tools.str_to_int(fig_name))
        fig.clf()
    num_states = feature.shape[1]
    for state_index in range(num_states):
        feature_sensors = feature[:,state_index]
        # Maximize contrast
        feature_sensors *= 1 / (np.max(feature_sensors) + tools.EPSILON)
        pixel_values = ((feature_sensors[ 0:n_pixels] - 
                         feature_sensors[n_pixels:2 * n_pixels]) + 1.0) / 2.0
        feature_pixels = pixel_values.reshape(fov_vert_span, fov_horz_span)
        if array_only:
            pixel_array.append(feature_pixels)
        else:
            plt.gray()
            ax = fig.add_axes((float(state_index)/float(num_states), 0., 
                               1/float(num_states), 1.), frame_on=False)
            im = plt.imshow(feature_pixels, vmin=0.0, vmax=1.0, 
                            interpolation='nearest')
            plt.title(fig_title)
    if array_only:
        return pixel_array
    else:
        if save_png:
            filename = (filename + '_' + world_name  + '_' + block_str + 
                        '_' + feature_str + '.png')
            fig.savefig(filename, format='png')
        fig.show()
        fig.canvas.draw()
        return

def print_pixel_array_features(projections, num_sensors, num_actions, 
                               fov_horz_span, fov_vert_span, 
                               directory='log', world_name=''):
    num_blocks = len(projections)
    for block_index in range(num_blocks):
        for feature_index in range(len(projections[block_index])):
            states_per_feature = block_index + 2
            plt.close(99)
            feature_fig = plt.figure(num=99)
            projection_image_list = (visualize_pixel_array_feature(projections[
                    block_index][feature_index][
                    num_actions:num_actions + num_sensors,:], fov_horz_span,
                    fov_vert_span, array_only=True)) 
            for state_index in range(states_per_feature): 
                left =  (float(state_index) / float(states_per_feature))
                bottom = 0.
                width =  1. /  float(states_per_feature)
                height =  1
                rect = (left, bottom, width, height)
                ax = feature_fig.add_axes(rect)
                plt.gray()
                ax.imshow(projection_image_list[state_index], 
                          interpolation='nearest', vmin=0., vmax=1.)
            # create a plot of individual features
            filename = '_'.join(('block', str(block_index).zfill(2),
                                 'feature',str(feature_index).zfill(4),
                                 world_name, 'world.png'))
            full_filename = os.path.join(directory, filename)
            plt.title(filename)
            plt.savefig(full_filename, format='png') 
    return

def make_movie(stills_directory, movie_filename='', frames_per_still = 1):
    if not movie_filename:
        movie_filename = ''.join((stills_directory, '.avi'))
    stills_filenames = []
    extensions = ['.png', '.jpg']
    stills_filenames = tools.get_files_with_suffix(stills_directory, extensions)
    stills_filenames.sort()
    print 'mf', movie_filename
    print 'st', len(stills_filenames)

    image = cv2.imread(stills_filenames[0])
    (height, width, depth) = image.shape
    frame_size = (width, height)
    """ fourCC code for the encoder to use"""
    codec = 'MJPG' # pretty good quality, claims to be broadly supported
    fourcc = cv2.cv.CV_FOURCC(codec[0], codec[1], codec[2], codec[3])
    fps = 30.
    is_color = True
    video_writer = cv2.VideoWriter(movie_filename, fourcc, fps, 
                                   frame_size, is_color)
    for filename in stills_filenames:
        print 'writing', filename
        image = cv2.imread(filename)
        resized_image = resample2D(image, height, width)
        for frame_counter in range(frames_per_still):
            video_writer.write(resized_image)

def resample2D(array, num_rows, num_cols):
    """ Return resampled array that is num_rows by num_cols """
    rows = (np.linspace(0., .9999999, num_rows) * 
            array.shape[0]).astype(np.int) 
    cols = (np.linspace(0., .9999999, num_cols) * 
            array.shape[1]).astype(np.int) 
    resampled_array = array[rows, :,:]
    resampled_array = resampled_array[:, cols,:]
    return resampled_array
