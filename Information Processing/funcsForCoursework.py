import numpy as np
import matplotlib.pyplot as plt
import PIL as pl

plt.style.use('default')
import math


def calcNCC(A, B):
    """
  function to calculate the normalised cross correlation between
  two images
  
  INPUTS:    A: an image stored as a 2D matrix
             B: an image stored as a 2D matrix. B must the the same size as
                 A
  
  OUTPUTS:   NCC: the value of the normalised cross correlation
  
  NOTE: if either of the images contain NaN values these pixels should be
  ignored when calculating the SSD.

  """
    # use nanmean and nanstd functions to calculate mean and std dev of each
    # image
    mu_A = np.nanmean(A)
    mu_B = np.nanmean(B)
    sig_A = np.nanstd(A, ddof=1)
    sig_B = np.nanstd(B, ddof=1)
    # calculate NCC using nansum to ignore nan values when summing over pixels
    return np.nansum((A - mu_A) * (B - mu_B)) / (A.size * sig_A * sig_B)


def dispImage(img, int_lims=[], ax=None):
    """
  function to display a grey-scale image that is stored in 'standard
  orientation' with y-axis on the 2nd dimension and 0 at the bottom

  INPUTS:   img: image to be displayed
            int_lims: the intensity limits to use when displaying the
               image, int_lims(1) = min intensity to display, int_lims(2)
               = max intensity to display [default min and max intensity
               of image]
            ax: if displaying an image on a subplot grid or on top of a
              second image, optionally supply the axis on which to display 
              the image.
  """

    # check if intensity limits have been provided, and if not set to min and
    # max of image
    if not int_lims:
        int_lims = [np.nanmin(img), np.nanmax(img)]
        # check if min and max are same (i.e. all values in img are equal)
        if int_lims[0] == int_lims[1]:
            # add one to int_lims(2) and subtract one from int_lims(1), so that
            # int_lims(2) is larger than int_lims(1) as required by imagesc
            # function
            int_lims[0] -= 1
            int_lims[1] += 1
    # take transpose of image to switch x and y dimensions and display with
    # first pixel having coordinates 0,0
    img = img.T
    if not ax:
        plt.imshow(img, cmap='gray', vmin=int_lims[0], vmax=int_lims[1], \
                   origin='lower')
    else:
        ax.imshow(img, cmap='gray', vmin=int_lims[0], vmax=int_lims[1], \
                  origin='lower')
    # set axis to be scaled equally (assumes isotropic pixel dimensions), tight
    # around the image
    plt.axis('image')
    plt.tight_layout()
    return ax


def resampImageWithDefField(source_img, def_field, interp_method='linear'):
    """
  function to resample a 2D image with a 2D deformation field

  INPUTS:    source_img: the source image to be resampled, as a 2D matrix
             def_field: the deformation field, as a 3D matrix
             inter_method: any of the interpolation methods accepted by
                 interpn function [default = 'linear'] - 
                 'linear', 'nearest' and 'splinef2d'
  OUTPUTS:   resamp_img: the resampled image
  
  NOTES: the deformation field should be a 3D numpy array, where the size of the
  first two dimensions is the size of the resampled image, and the size of
  the 3rd dimension is 2. def_field[:,:,0] contains the x coordinates of the
  transformed pixels, def_field[:,:,1] contains the y coordinates of the
  transformed pixels.
  the origin of the source image is assumed to be the bottom left pixel
  """
    x_coords = np.arange(source_img.shape[0], dtype='float')
    y_coords = np.arange(source_img.shape[1], dtype='float')
    from scipy.interpolate import interpn
    # resample image using interpn function
    return interpn((x_coords, y_coords), source_img, def_field, bounds_error=False, \
                   fill_value=np.NAN, method=interp_method)


def dispDefField(def_field, ax=None, spacing=5, disptype='grid'):
    """
  function to display a deformation field
  
  INPUTS:    def_field: the deformation field as a 3D array
             ax: the axis on which to plot the deformation field
             spacing: the spacing of the grids/arrows in pixels [5]
             type: the type of display to use, 'grid' or 'arrows' ['grid']
  """
    if not ax:
        ax = plt.subplot(111)

    if disptype == 'grid':
        # plot vertical grid lines
        for i in np.arange(0, def_field.shape[0], spacing):
            ax.plot(def_field[i, :, 0], def_field[i, :, 1], c='red', linewidth=0.5)
        # plot horizontal grid lines
        for j in np.arange(0, def_field.shape[1], spacing):
            ax.plot(def_field[:, j, 0], def_field[:, j, 1], c='red', linewidth=0.5)

    else:
        if disptype == 'arrows':
            # calculate displacement field from deformation field
            X, Y = np.mgrid[0:def_field.shape[0], 0:def_field.shape[1]]
            arrow_disp_x = def_field[::spacing, ::spacing, 0] - X[::spacing, ::spacing]
            arrow_disp_y = def_field[::spacing, ::spacing, 1] - Y[::spacing, ::spacing]
            M = np.hypot(arrow_disp_x, arrow_disp_y)
            # plot displacements using quiver
            ax.quiver(X[::spacing, ::spacing], Y[::spacing, ::spacing], \
                      arrow_disp_x, arrow_disp_y, M, scale=1.0, angles='xy', scale_units='xy', \
                      cmap='jet', headwidth=2, width=0.004, headlength=3)
        else:
            print('Display type must be grid or arrows')
    # set the axis limits and appearance
    ax.axis('image')
    ax.set_xlim([-1, def_field.shape[0]])
    ax.set_ylim([-1, def_field.shape[1]])
    return ax


def read_file(filename):
    imagefile = pl.Image.open(filename)
    image_array = np.array(imagefile.getdata(), np.uint64).reshape(imagefile.size[1], imagefile.size[0])
    return image_array.astype('double')/255

def array_to_image(array):
    minimal_value = np.min(array)
    maximal_value = np.max(array)
    if minimal_value < 0 or maximal_value > 255:
        array = 255*(array-minimal_value)/(maximal_value-minimal_value)
    array_uint8 = array.astype('uint8')
    return pl.Image.fromarray(array_uint8, 'L') #saves as 8-bit pixels

## save array as image file
def save_file(array,filename):
    imagefile = array_to_image(array)
    imagefile.save(filename)
