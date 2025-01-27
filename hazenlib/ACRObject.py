import sys
import cv2
import scipy
import skimage
import numpy as np
from hazenlib.logger import logger
from hazenlib.utils import determine_orientation, detect_circle, detect_centroid, debug_image_sample, expand_data_range


class ACRObject:
    """Base class for performing tasks on image sets of the ACR phantom. \n
    acquired following the ACR Large phantom guidelines
    """

    def __init__(self, dcm_list):
        """Initialise an ACR object instance

        Args:
            dcm_list (list): list of pydicom.Dataset objects - DICOM files loaded
        """
        # First, need to determine if input DICOMs are
        # enhanced or normal, single or multi-frame
        # may be 11 in 1 or 11 separate DCM objects

        # # Initialise an ACR object from a list of images of the ACR phantom
        # Store pixel spacing value from the first image (expected to be the same for all)
        self.dx, self.dy = dcm_list[0].PixelSpacing

        # Perform sorting of the input DICOM list based on position
        sorted_dcms = self.sort_dcms(dcm_list)

        # Perform sorting of the image slices based on phantom orientation
        self.slice_stack = self.order_phantom_slices(sorted_dcms)
        logger.info(f'Ordered slices => {[sl.InstanceNumber for sl in self.slice_stack]}')

    def sort_dcms(self, dcm_list):
        """Sort a stack of DICOM images based on slice position.

        Args:
            dcm_list (list): list of pyDICOM image objects

        Returns:
            list: sorted list of pydicom.Dataset objects
        """
        orientation, positions = determine_orientation(dcm_list)
        if orientation == "unexpected":
            # TODO: error out for now,
            # in future allow manual override based on optional CLI args
            logger.error(f'Unknown orientation detected => {orientation}')
            sys.exit()

        logger.info("image orientation is %s", orientation)
        dcm_stack = [dcm_list[i] for i in np.argsort(positions)]
        # img_stack = [dcm.pixel_array for dcm in dcm_stack]
        return dcm_stack  # , img_stack

    def order_phantom_slices(self, dcm_list):
        """Determine slice order based on the detection of the small circle in the first slice
        # or an LR orientation swap is required. \n

        # This function analyzes the given set of images and their associated DICOM objects to determine if any
        # adjustments are needed to restore the correct slice order and view orientation.

        Args:
            dcm_list (list): list of pyDICOM image objects

        Returns:
            list: sorted list of pydicom.Dataset objects corresponding to ordered phantom slices
        """
        # Check whether the circle is on the first or last slice

        # Get pixel array of first and last slice
        first_slice = dcm_list[0].pixel_array
        last_slice = dcm_list[-1].pixel_array
        # Detect circles in the first and last slice
        detected_circles_first = detect_circle(first_slice, self.dx)
        detected_circles_last = detect_circle(last_slice, self.dx)

        # It is assumed that only the first or the last slice has circles
        if detected_circles_first is not None and detected_circles_last is None:
            # If first slice has the circle then slice order is correct
            logger.info("Slice order inversion is not required.")
            return dcm_list
        if detected_circles_first is None and detected_circles_last is not None:
            # If last slice has the circle then slice order needs to be reversed
            logger.info("Performing slice order inversion.")
            return dcm_list[::-1]

        logger.debug("Neither slices had a circle detected")
        return dcm_list

    @staticmethod
    def determine_rotation(img):
        """Determine the rotation angle of the phantom using edge detection and the Hough transform.
        only relevant for MTF-based spatial resolution - need to convince David Price!!!!!

        Args:
            img (np.ndarray): pixel array of a DICOM object

        Returns:
            float: The rotation angle in degrees.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
        diff = cv2.absdiff(dilate, thresh)

        h, theta, d = skimage.transform.hough_line(diff)
        _, angles, _ = skimage.transform.hough_line_peaks(h, theta, d)

        if len(angles) >= 1:
            angle = np.rad2deg(scipy.stats.mode(angles, keepdims=False).mode)
            if angle < 0:
                rot_angle = angle + 90
            else:
                rot_angle = angle - 90
        else:
            rot_angle = 0
        logger.info("Phantom is rotated by %s degrees", rot_angle)

        return rot_angle

    def rotate_images(self, dcm_list, rot_angle):
        """Rotate the images by a specified angle. The value range and dimensions of the image are preserved.

        Args:
            dcm_list (list): list of pyDICOM image objects
            rot_angle (float): angle in degrees that image (pixel array) should be rotated by

        Returns:
            list of np.ndarray: The rotated images.
        """
        rotated_images = skimage.transform.rotate(
            dcm_list, rot_angle, resize=False, preserve_range=True
        )
        return rotated_images

    @staticmethod
    def find_phantom_center(img, dx, dy):
        """Find the center of the ACR phantom in a given slice (pixel array) \n
        using the Hough circle detector on a blurred image.

        Args:
            img (np.ndarray): pixel array of the DICOM image.
            dx (int): pixel array of the DICOM image.
            dy (int): pixel array of the DICOM image.

        Returns:
            tuple of ints: (x, y) coordinates of the center of the image
        """
        logger.info("Detecting centroid location ...")
        detected_circles = detect_centroid(img, dx, dy)
        centre_x = round(detected_circles[0])
        centre_y = round(detected_circles[1])
        radius = round(detected_circles[2])
        logger.info(f"Centroid (x, y) => {centre_x}, {centre_y}")

        return (centre_x, centre_y), radius

    def get_mask_image(self, image, centre, mag_threshold=0.07, open_threshold=500):
        """Create a masked pixel array. \n
        Mask an image by magnitude threshold before applying morphological opening to remove small unconnected
        features. The convex hull is calculated in order to accommodate for potential air bubbles.

        Args:
            image (np.ndarray): pixel array of the dicom
            centre (tuple): x,y coordinates of the circle centre.
            mag_threshold (float, optional): magnitude threshold. Defaults to 0.07.
            open_threshold (int, optional): open threshold. Defaults to 500.

        Returns:
            np.ndarray: the masked image
        """
        test_mask = self.circular_mask(centre, (80 // self.dx), image.shape)
        test_image = image * test_mask
        # get range of values in the mask
        test_vals = test_image[np.nonzero(test_image)]
        if np.percentile(test_vals, 80) - np.percentile(test_vals, 10) > 0.9 * np.max(
            image
        ):
            print(
                "Large intensity variations detected in image. Using local thresholding!"
            )
            initial_mask = skimage.filters.threshold_sauvola(
                image, window_size=3, k=0.95
            )
        else:
            initial_mask = image > mag_threshold * np.max(image)

        # unconnected region of pixels, to remove noise
        opened_mask = skimage.morphology.area_opening(
            initial_mask, area_threshold=open_threshold
        )
        # remove air bubbles from image area
        final_mask = skimage.morphology.convex_hull_image(opened_mask)

        return final_mask

    @staticmethod
    def circular_mask(centre, radius, dims):
        """Generates a circular mask using given centre coordinates and a given radius. Generates a linspace grid the
        size of the given dimensions and checks whether each point on the linspace grid is within the desired radius
        from the given centre coordinates. Each linspace value within the chosen radius then becomes part of the mask.


        Args:
            centre (tuple): centre coordinates of the circular mask.
            radius (int): radius of the circular mask.
            dims (tuple): dimensions to create the base linspace grid from.

        Returns:
            np.ndarray: A sorted stack of images, where each image is represented as a 2D numpy array.
        """
        # Define a circular logical mask
        x = np.linspace(1, dims[0], dims[0])
        y = np.linspace(1, dims[1], dims[1])

        X, Y = np.meshgrid(x, y)
        mask = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 <= radius**2

        return mask

    def measure_orthogonal_lengths(self, mask, cxy):
        """Compute the horizontal and vertical lengths of a mask, based on the centroid.

        Args:
            mask (np.ndarray): Boolean array of the image where pixel values meet threshold
            cxy  (tuple): x,y coordinates of the circle centre.

        Returns:
            dict: a dictionary with the following:
                'Horizontal Start'      | 'Vertical Start' : tuple of int
                    Horizontal/vertical starting point of the object.
                'Horizontal End'        | 'Vertical End' : tuple of int
                    Horizontal/vertical ending point of the object.
                'Horizontal Extent'     | 'Vertical Extent' : np.ndarray of int
                    Indices of the non-zero elements of the horizontal/vertical line profile.
                'Horizontal Distance'   | 'Vertical Distance' : float
                    The horizontal/vertical length of the object.
        """
        dims = mask.shape
        (vertical, horizontal) = cxy

        horizontal_start = (horizontal, 0)
        horizontal_end = (horizontal, dims[0] - 1)
        horizontal_line_profile = skimage.measure.profile_line(
            mask, horizontal_start, horizontal_end
        )
        horizontal_extent = np.nonzero(horizontal_line_profile)[0]
        horizontal_distance = (horizontal_extent[-1] - horizontal_extent[0]) * self.dx

        vertical_start = (0, vertical)
        vertical_end = (dims[1] - 1, vertical)
        vertical_line_profile = skimage.measure.profile_line(
            mask, vertical_start, vertical_end
        )
        vertical_extent = np.nonzero(vertical_line_profile)[0]
        vertical_distance = (vertical_extent[-1] - vertical_extent[0]) * self.dy

        length_dict = {
            "Horizontal Start": horizontal_start,
            "Horizontal End": horizontal_end,
            "Horizontal Extent": horizontal_extent,
            "Horizontal Distance": horizontal_distance,
            "Vertical Start": vertical_start,
            "Vertical End": vertical_end,
            "Vertical Extent": vertical_extent,
            "Vertical Distance": vertical_distance,
        }

        return length_dict

    @staticmethod
    def rotate_point(origin, point, angle):
        """Compute the horizontal and vertical lengths of a mask, based on the centroid.

        Args:
            origin (tuple): The coordinates of the point around which the rotation is performed.
            point (tuple): The coordinates of the point to rotate.
            angle (int): Angle in degrees.

        Returns:
            tuple of float: Floats representing the x and y coordinates of the input point
            after being rotated around an origin.
        """
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)

        x_prime = origin[0] + c * (point[0] - origin[0]) - s * (point[1] - origin[1])
        y_prime = origin[1] + s * (point[0] - origin[0]) + c * (point[1] - origin[1])
        return x_prime, y_prime

    @staticmethod
    def find_n_highest_peaks(data, n, height=1):
        """Find the indices and amplitudes of the N highest peaks within a 1D array.

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            n (int): The coordinates of the point to rotate
            height (int, optional): The amplitude threshold for peak identification. Defaults to 1.

        Returns:
            tuple of np.ndarray:
                peak_locs: A numpy array containing the indices of the N highest peaks identified. \n
                peak_heights: A numpy array containing the amplitudes of the N highest peaks identified.

        """
        peaks = scipy.signal.find_peaks(data, height)
        pk_heights = peaks[1]["peak_heights"]
        pk_ind = peaks[0]

        peak_heights = pk_heights[
            (-pk_heights).argsort()[:n]
        ]  # find n highest peak amplitudes
        peak_locs = pk_ind[(-pk_heights).argsort()[:n]]  # find n highest peak locations

        return np.sort(peak_locs), np.sort(peak_heights)

    @staticmethod
    def apply_window_width_center(data, center, width):
        """Filters data by the specified center and width.

        ::

            Murphy A, Wilczek M, Feger J, et al. Windowing (CT). Reference article,
            Radiopaedia.org (Accessed on 24 Jan 2025) https://doi.org/10.53347/rID-52108

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            center (int): The desired Window Center setting.
            width (int): The desired Window Width setting.

        Returns:
            np.ndarray: Scaled data

        """
        dtype = data.dtype
        dtype_max = np.iinfo(dtype).max
        half_width = width / 2
        upper_grey = center + half_width
        lower_grey = center - half_width
        logger.info(f'Applying Window Settings from => Center: {center} Width: {width}')
        logger.info(f'Window Thresholds => Upper Threshold: {upper_grey} Lower Threshold: {lower_grey}')
        upper_mask = data > upper_grey
        lower_mask = data < lower_grey
        mid_mask = upper_mask & lower_mask
        masked_data = np.ma.masked_array(data.copy(), mask=mid_mask, fill_value=0)
        # Apply thresholds
        masked_data[lower_mask] = 0
        masked_data[upper_mask] = dtype_max
        # Stretch center values across the data type range
        return expand_data_range(masked_data, (lower_grey, upper_grey), dtype)

    @staticmethod
    def compute_width_and_center(data):
        """Automatically resolves the center and width settings from the given data. If you wish to derive the
        center and width from roi

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data to perform center and width calculation

        Returns:
            tuple of int:
                center (float): The desired Window Center setting. \n
                width (float): The desired Window Width setting.

        """
        width = np.std(data)
        return np.round(ACRObject.compute_data_mode(data)), np.round(width)

    @staticmethod
    def compute_data_mode(data):
        """Computes the mode of the given dataset using a 35 bins histogram. This method ignores the zeros.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data

        Returns:
            mode (float): non-zero mode of the dataset.

        """
        search_data = data[data > 0]
        hist, bins = np.histogram(search_data, bins=35)
        mode = bins[np.argmax(hist)]
        logger.info(f'Computed mode: {mode}')
        return mode

    @staticmethod
    def threshold_data(data, intensity, fill=0):
        """Thresholds the data. Meaning, every pixel with value < intensity will be replaced by the value in fill.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data
            intensity (float): pixel value to use as the threshold
            fill (float, optional): pixel value to use as replacement. Defaults to 0.

        Returns:
            mode (float): non-zero mode of the dataset.

        """
        data[data < intensity] = fill
        return data

