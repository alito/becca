"""
A few functions that are useful to mutliple worlds
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import core.tools as tools

def center_surround(fov, fov_horz_span, fov_vert_span, verbose=False):
    """ 
    Convert a 2D array of b/w pixel values to center-surround 
    
    Args:
        fov: field of view, 2D array of pixel values
        fov_horz_span: desired number of center-surround superpixel columns
        fov_vert_span: desired number of center-surround superpixel rows
    Returns: 
        a 2D array of the center surround values
    """ 
    fov_height = fov.shape[0]
    fov_width = fov.shape[1]
    block_width = float(fov_width) / float(fov_horz_span + 2)
    block_height = float(fov_height) / float(fov_vert_span + 2)
    super_pixels = np.zeros((fov_vert_span + 2, fov_horz_span + 2))
    center_surround_pixels = np.zeros((fov_vert_span, fov_horz_span))
    # Create the superpixels by averaging pixel blocks
    for row in range(fov_vert_span + 2):
        for col in range(fov_horz_span + 2):
            super_pixels[row][col] = np.mean(
                    fov[int(float(row)     * block_height): 
                        int(float(row + 1) * block_height),
                        int(float(col)     * block_width) : 
                        int(float(col + 1) * block_width) ])
    for row in range(fov_vert_span):
        for col in range(fov_horz_span):
            # Calculate a center-surround value that represents
            # the difference between the pixel and its surroundings.
            # Weight the N, S, E, and W pixels by 1/6 and
            # the NW, NE, SW, and SE pixels by 1/12, and 
            # subtract from the center.
            center_surround_pixels[row][col] = (
                    super_pixels[row + 1][col + 1] -
                    super_pixels[row    ][col + 1] / 6 -
                    super_pixels[row + 2][col + 1] / 6 -
                    super_pixels[row + 1][col    ] / 6 -
                    super_pixels[row + 1][col + 2] / 6 -
                    super_pixels[row    ][col    ] / 12 -
                    super_pixels[row + 2][col    ] / 12 -
                    super_pixels[row    ][col + 2] / 12 -
                    super_pixels[row + 2][col + 2] / 12)
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
    n_pixels = feature.shape[0] / 2
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

def print_pixel_array_features(projections, num_pixels_x2, start_index, 
                               fov_horz_span, fov_vert_span, 
                               directory='log', world_name='', name='',
                               interp='nearest'):
    """  Interpret an array of center-surround pixels as an image """
    num_blocks = len(projections)
    for block_index in range(num_blocks):
        for feature_index in range(len(projections[block_index])):
            states_per_feature = 2 ** (block_index + 1)
            plt.close(99)
            feature_fig = plt.figure(num=99)
            projection_image_list = (visualize_pixel_array_feature(projections[
                    block_index][feature_index][
                    start_index:start_index + num_pixels_x2,:], fov_horz_span,
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
                          interpolation=interp, vmin=0., vmax=1.)
            # create a plot of individual features
            filename = '_'.join(('block', str(block_index).zfill(2),
                                 'feature',str(feature_index).zfill(4),
                                 world_name, 'world', name, 'image.png'))
            full_filename = os.path.join(directory, filename)
            plt.title(filename)
            plt.savefig(full_filename, format='png') 

def make_movie(stills_directory, movie_filename='', frames_per_still=1,
               stills_per_frame=1):
    """ Make a movie out of a sequence of still frames """
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
    # MJPG is pretty good quality and claims to be broadly supported
    codec = 'MJPG' 
    fourcc = cv2.cv.CV_FOURCC(codec[0], codec[1], codec[2], codec[3])
    fps = 30.
    is_color = True
    video_writer = cv2.VideoWriter(movie_filename, fourcc, fps, 
                                   frame_size, is_color)
    images = []
    num_stills_this_frame = 0
    for filename in stills_filenames:
        print 'writing', filename
        image = cv2.imread(filename)
        resized_image = resample2D(image, height, width)
        images.append(resized_image)
        num_stills_this_frame += 1
        if num_stills_this_frame == stills_per_frame:
            image = np.zeros(images[0].shape)
            for image_ in images:
                image += image_
            image = (image / len(images)).astype('uint8')
            for frame_counter in range(frames_per_still):
                video_writer.write(image)
            num_stills_this_frame = 0
            images = []

def resample2D(array, num_rows, num_cols):
    """ Resample a 2D array to get one that has num_rows and num_cols """
    rows = (np.linspace(0., .9999999, num_rows) * 
            array.shape[0]).astype(np.int) 
    cols = (np.linspace(0., .9999999, num_cols) * 
            array.shape[1]).astype(np.int) 
    if len(array.shape) == 2:
        resampled_array = array[rows, :]
        resampled_array = resampled_array[:, cols]
    if len(array.shape) == 3:
        resampled_array = array[rows, :,:]
        resampled_array = resampled_array[:, cols,:]
    return resampled_array

def duration_string(time_in_sec):
    """ Convert time in seconds to a human readable date string """
    sec_per_min = 60
    min_per_hr = 60
    hr_per_day = 24
    day_per_mon = 30
    mon_per_yr = 12
    # Calculate seconds
    #print 'tis', time_in_sec
    sec = time_in_sec - sec_per_min * int(time_in_sec / sec_per_min)
    time_in_min = (time_in_sec - sec) / sec_per_min
    duration = ''.join(('%0.2f' % (sec), 's'))
    #print 's', duration
    if time_in_min == 0:
        return duration
    # Calculate minutes
    min = time_in_min - min_per_hr * int(time_in_min / min_per_hr)
    time_in_hr = (time_in_min - min) / min_per_hr
    duration = ''.join((str(int(min)), 'm ', duration))
    #print 'm', duration
    if time_in_hr == 0:
        return duration
    # Calculate hours
    hr = time_in_hr - hr_per_day * int(time_in_hr / hr_per_day)
    time_in_day = (time_in_hr - hr) / hr_per_day
    duration = ''.join((str(int(hr)), 'h ', duration))
    #print 'h', duration
    if time_in_day == 0:
        return duration
    # Calculate days
    day = time_in_day - day_per_mon * int(time_in_day / day_per_mon)
    time_in_mon = (time_in_day - day) / day_per_mon
    duration = ''.join((str(int(day)), 'd ', duration))
    #print 'd', duration
    if time_in_mon == 0:
        return duration
    # Calculate months
    mon = time_in_mon - mon_per_yr * int(time_in_mon / mon_per_yr)
    time_in_yr = (time_in_mon - mon) / mon_per_yr
    duration = ''.join((str(int(mon)), 'l ', duration))
    #print 'd', duration
    if time_in_yr == 0:
        return duration
    else:
        duration = ''.join((str(int(time_in_yr)), 'y ', duration))
        return duration

    print 'dont reach', 'dur', duration
