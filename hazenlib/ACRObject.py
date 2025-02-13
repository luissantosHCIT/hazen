import sys
import cv2
import scipy
import skimage
import numpy as np
from hazenlib.logger import logger
from hazenlib.utils import determine_orientation, detect_circle, detect_centroid, debug_image_sample, expand_data_range, \
    create_circular_kernel, get_datatype_min, get_datatype_max


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
        self.dx, self.dy = float(self.dx), float(self.dy)
        logger.info(f'In-plane acquisition resolution is {self.dx} x {self.dy}')

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
            dx (float): pixel array of the DICOM image.
            dy (float): pixel array of the DICOM image.

        Returns:
            tuple of ints: (x, y) coordinates of the center of the image
        """
        logger.info("Detecting centroid location ...")
        detected_circles = detect_centroid(img, dx, dy)
        centre_x = int(np.round(detected_circles[1]))
        centre_y = int(np.round(detected_circles[0]))
        radius = np.round(detected_circles[2])
        logger.info(f"Centroid (x, y) => {centre_x}, {centre_y}")

        return (centre_x, centre_y), radius

    @staticmethod
    def get_presentation_pixels(dcm):
        """Automatically resolves the pixel values as would have been expected for presentation.
        We follow section `C.11.2.1.2.2 General Requirements for Window Center and Window Width <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.3>`_
        from the DICOM standard.

        This means we do:

            #. Look for expected windowing function as per section C.11.2.1.3 VOI LUT Function. We default to LINEAR as
                the LUT Function if no other function is defined.
            #. Look for Window Center (0028,1050), Window Width (0028,1051) and VOI LUT Function (0028,1056) Attributes.
                If VOI LUT and Window settings are present, I currently default to the Window Settings and assume the
                LUT is an alternative or secondary display setting.
            #. Find max value for pixel array data type.
            #. Cast initial pixel array to float64 for calculation.
            #. Apply base rescaling of pixel data.
            #. Apply window function (default is linear) with default window settings.
            #. Return transformed data.

        .. note::

            #. For the purpose of this definition, a floating point calculation without integer truncation is assumed,
                though the manner of implementation may vary as long as the result is the same.
            #. The pseudo-code function computes a continuous value over the output range without any discontinuity at
                the boundaries. The value of 0 for w is expressly forbidden, and the value of 1 for w does not cause
                division by zero, since the continuous segment of the function will never be reached for that case.

            #. For example, for an output range 0 to 255:

                c=2048, w=4096 becomes:

                    if (x <= 0) then y = 0

                    else if (x > 4095) then y = 255

                    else y = ((x - 2047.5) / 4095 + 0.5) * (255-0) + 0

                c=2048, w=1 becomes:

                    if (x <= 2047.5) then y = 0

                    else if (x > 2047.5) then y = 255

                    else /* not reached */

                c=0, w=100 becomes:

                    if (x <= -50) then y = 0

                    else if (x > 49) then y = 255

                    else y = ((x + 0.5) / 99 + 0.5) * (255-0) + 0

                c=0, w=1 becomes:

                    if (x <= -0.5) then y = 0

                    else if (x > -0.5) then y = 255

                    else /* not reached */

            #. A Window Center of 2n-1 and a Window Width of 2n selects the range of input values from 0 to 2n-1.
                This represents a mathematical identity VOI LUT transformation over the possible input values
                (whether used or not) in the case where no Modality LUT is specified and the stored pixel data are n
                bit unsigned integers.

            #. In the case where x1 is the lowest input value actually used in the Pixel Data and x2 is the highest,
                a Window Center of (x1+x2+1)/2 and a Window Width of (x2-x1+1) selects the range of input values from
                x1 to x2, which represents the full range of input values present as opposed to possible. This is
                distinct from the mathematical identity VOI LUT transformation, which instead selects the full range
                of input values possible as opposed to those actually used. The mathematical identity and full input
                range transformations are the same when x1 = 0 and x2 is 2n-1 and the input values are n bit unsigned
                integers. See also Note 7.

            #. A Window Width of 1 is typically used to represent a "threshold" operation in which those integer
                input values less than the Window Center are represented as the minimum displayed value and those
                greater than or equal to the Window Center are represented as the maximum displayed value. A Window
                Width of 2 will have the same result for integral input values.

            #. The application of Window Center (0028,1050) and Window Width (0028,1051) may select a signed input
                range. There is no implication that this signed input range is clipped to zero.

            #. The selected input range may exceed the actual range of the input values, thus effectively "compressing"
                the contrast range of the displayed data into a narrower band of the available contrast range, and
                "flattening" the appearance. There are no limits to the maximum value of the window width, or to the
                minimum or maximum value of window level, both of which may exceed the actual or possible range of
                input values.

            #. Input values "below" the window are displayed as the minimum output value and input values "above" the
                window are displayed as the maximum output value. This is the common usage of the window operation
                in medical imaging. There is no provision for an alternative approach in which all values "outside"
                the window are displayed as the minimum output value.

            #. The output of the Window Center/Width or VOI LUT transformation is either implicitly scaled to the
                full range of the display device if there is no succeeding transformation defined, or implicitly scaled to
                the full input range of the succeeding transformation step (such as the Presentation LUT), if present.
                See Section C.11.6.1.

            #. Fractional values of Window Center and Window Width are permitted (since the VR of these Attributes is
                Decimal String), and though they are not often encountered, applications should be prepared to accept
                them.

        .. warning::

            I do not implement all aspects of section C.11.2.1.2. Any elements that are found as needed in the wild
            should be retrospectively implemented.

        Args:
            dcm (pydicom.Dataset): DICOM instance used to find the default windowing and rescaling settings.

        Returns:
            raw (np.ndarray): Original pixel array.
            rescaled (np.ndarray): Rescaled pixel array.
            presentation (np.ndarray): Presentation ready pixel array.

        """
        img = dcm.pixel_array
        dtype = img.dtype
        slope = dcm.get('RescaleSlope', 1)
        intercept = dcm.get('RescaleIntercept', 1)
        center = dcm.get('WindowCenter', None)
        width = dcm.get('WindowWidth', None)
        voi_lut_function = dcm.get('VOILUTFunction', 'linear').lower()
        float_data = img.copy().astype(np.float32)  # Cast to float to maintain precision
        rescaled = ACRObject.rescale_data(float_data, slope, intercept)
        windowed = ACRObject.apply_window_center_width(rescaled, center, width, voi_lut_function, dtype)
        rounded_window = np.round(windowed.data)    # Round so that we have integers. Realistically, we should only be
                                                    # dealing with uint16, but adjust this step if there's other data.
        return img, np.round(rescaled).astype(dtype), rounded_window.astype(dtype)  # Cast back to original type, allow truncation

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
    def rescale_data(data, slope=1, intercept=0):
        """Rescales the data by a given slope and intercept.

        The equation is y = mx + b

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            slope (float): The slope m.
            intercept (float): The intercept b.

        Returns:
            np.ndarray: Scaled data
        """
        return np.add(np.multiply(slope, data), intercept)

    @staticmethod
    def apply_linear_window_center_width(data, center, width, dtmin=0, dtmax=255):
        """Filters data by the specified center and width using the DICOM linear equation.
        See `C.11.2.1.2.1 Default LINEAR Function <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2.1>`_.

        ::

            if (x <= c - 0.5 - (w-1) /2), then y = ymin
            else if (x > c - 0.5 + (w-1) /2), then y = ymax
            else y = ((x - (c - 0.5)) / (w-1) + 0.5) * (ymax- ymin) + ymin

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            center (int): The desired Window Center setting.
            width (int): The desired Window Width setting.
            dtmin (int): The min value of datatype.
            dtmax (int): The max value of datatype.

        Returns:
            np.ma.MaskedArray: Windowed data
        """
        logger.info(f'Applying Window Settings using the Linear method => Center: {center} Width: {width}')
        adjusted_width = width - 1
        half_width = adjusted_width / 2
        lower_mask = data <= center - 0.5 - half_width
        upper_mask = data > center - 0.5 + half_width
        mid_mask = lower_mask & upper_mask
        masked_data = np.ma.masked_array(data.copy(), mask=mid_mask, fill_value=0)
        # Apply thresholds
        masked_data[lower_mask] = dtmin
        masked_data[upper_mask] = dtmax
        masked_data[~mid_mask] = ((masked_data[~mid_mask] - (center - 0.5)) / adjusted_width + 0.5) * (dtmax - dtmin) + dtmin
        return masked_data

    @staticmethod
    def apply_linear_exact_window_center_width(data, center, width, dtmin=0, dtmax=255):
        """Filters data by the specified center and width using the DICOM linear exact equation.
        See `C.11.2.1.2.1 LINEAR_EXACT Function <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.3.2>`_.

        ::

            if (x <= c - w/2), then y = ymin
            else if (x > c + w/2), then y = ymax
            else y = ((x - c) / w + 0.5) * (ymax- ymin) + ymin

        This equation is similar to described in Radiopaedia

        ::

            Murphy A, Wilczek M, Feger J, et al. Windowing (CT). Reference article,
            Radiopaedia.org (Accessed on 24 Jan 2025) https://doi.org/10.53347/rID-52108

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            center (int): The desired Window Center setting.
            width (int): The desired Window Width setting.
            dtmin (int): The min value of datatype.
            dtmax (int): The max value of datatype.

        Returns:
            np.ma.MaskedArray: Windowed data
        """
        logger.info(f'Applying Window Settings using the Linear Exact method => Center: {center} Width: {width}')
        half_width = width / 2
        lower_mask = data <= center - half_width
        upper_mask = data > center + half_width
        mid_mask = lower_mask & upper_mask
        masked_data = np.ma.masked_array(data.copy(), mask=mid_mask, fill_value=0)
        # Apply thresholds
        masked_data[lower_mask] = dtmin
        masked_data[upper_mask] = dtmax
        masked_data[~mid_mask] = ((masked_data[~mid_mask] - center) / width + 0.5) * (dtmax - dtmin) + dtmin
        return masked_data

    @staticmethod
    def apply_sigmoid_window_center_width(data, center, width, dtmin=0, dtmax=255):
        """Filters data by the specified center and width using the DICOM sigmoid equation.
        See `C.11.2.1.2.1 SIGMOID Function <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.3.1>`_.

        ::

            y = ((ymax - ymin) / (1 + exp(-4 * (x - c) / w))) + ymin

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            center (int): The desired Window Center setting.
            width (int): The desired Window Width setting.
            dtmin (int): The min value of datatype.
            dtmax (int): The max value of datatype.

        Returns:
            np.ma.MaskedArray: Windowed data
        """
        logger.info(f'Applying Window Settings using the Sigmoid method => Center: {center} Width: {width}')
        mid_mask = dtmin <= data <= dtmax
        masked_data = np.ma.masked_array(data.copy(), mask=mid_mask, fill_value=0)
        # Apply thresholds
        masked_data[~mid_mask] = ((dtmax - dtmin) / (1 + np.exp((-4 * masked_data[~mid_mask] - center) / width))) + dtmin
        return masked_data

    @staticmethod
    def apply_clip_window_center_width(data, center, width, dtmin=0, dtmax=255):
        """Filters data by the specified center and width using the custom clip method. This method is not a standard
        DICOM windowing method. It is a modified form of the linear_exact method. We basically do not rescale the window
        data. It uses numpy.clip() to generate the window data.

        Somehow, this works better than the linear method for the object detectability task.

        This method is similar to described in Radiopaedia

        ::

            Murphy A, Wilczek M, Feger J, et al. Windowing (CT). Reference article,
            Radiopaedia.org (Accessed on 24 Jan 2025) https://doi.org/10.53347/rID-52108

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            center (int): The desired Window Center setting.
            width (int): The desired Window Width setting.
            dtmin (int): The min value of datatype.
            dtmax (int): The max value of datatype.

        Returns:
            np.ma.MaskedArray: Windowed data
        """
        logger.info(f'Applying Window Settings using the Clip method => Center: {center} Width: {width}')
        half_width = width / 2
        upper_grey = center + half_width
        lower_grey = center - half_width
        upper_mask = data > upper_grey
        lower_mask = data <= lower_grey
        mid_mask = lower_mask & upper_mask
        masked_data = np.ma.masked_array(data.copy(), mask=mid_mask, fill_value=0)
        # Apply thresholds
        masked_data[lower_mask] = dtmin
        masked_data[upper_mask] = dtmax
        masked_data[~mid_mask] = np.clip(masked_data[~mid_mask], lower_grey, upper_grey)
        return masked_data

    @staticmethod
    def apply_window_center_width(data, center, width, function='linear', dtype=None):
        """Filters data by the specified center and width. We support 3 functions defined by the DICOM standard. These
        functions are linear (default), linear_exact, and sigmoid.

        See `DICOM Chapter 11.2 <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html>`_
        for more details.

        Each specific function is defined as apply_<function>_window_center_width and each has a copy of the DICOM
        documentation. See those for details as well.

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            center (int): The desired Window Center setting.
            width (int): The desired Window Width setting.
            function (str): Function t apply to data windowing. Defaults to linear.
            dtype (np.dtype): The data's natural datatype. We use the input's datatype if no specific data type is
                specified. This parameter is mainly present for other higher level methods to use to maintain
                logical consistency during processing.

        Returns:
            np.ma.MaskedArray: Windowed data.

        """
        dtype = data.dtype if dtype is None else dtype
        dtype_min = get_datatype_min(dtype)
        dtype_max = get_datatype_max(dtype)
        if function == 'linear_exact':
            return ACRObject.apply_linear_exact_window_center_width(data, center, width, dtype_min, dtype_max)
        if function == 'sigmoid':
            return ACRObject.apply_sigmoid_window_center_width(data, center, width, dtype_min, dtype_max)
        if function == 'clip':
            return ACRObject.apply_clip_window_center_width(data, center, width, dtype_min, dtype_max)
        return ACRObject.apply_linear_window_center_width(data, center, width, dtype_min, dtype_max)

    @staticmethod
    def compute_center_and_width(data):
        """Automatically resolves the center and width settings from the given data. If you wish to derive the
        center and width from roi

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data to perform center and width calculation

        Returns:
            tuple of int:
                center (float): The desired Window Center setting. \n
                width (float): The desired Window Width setting.

        """
        return np.round(ACRObject.compute_histogram_mean(data)), np.round(ACRObject.compute_histogram_width(data))

    @staticmethod
    def compute_histogram_mode(data):
        """Computes the mode of the given dataset using a 256 bins histogram. This method ignores the zeros.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data

        Returns:
            mode (float): non-zero mode of the dataset.

        """
        search_data = data[data > 0]
        hist, bins = np.histogram(search_data, bins=256)
        mode = bins[np.argmax(hist)]
        logger.info(f'Computed mode: {mode}')
        return mode

    @staticmethod
    def compute_histogram_mean(data):
        """Computes the mean of the given dataset using a 256 bins histogram. This method ignores the zeros.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data

        Returns:
            mean (float): non-zero mean of the dataset.

        """
        search_data = data[data > 0]
        hist, bins = np.histogram(search_data, bins=256)
        logger.info(np.mean(bins))
        mean = np.mean(bins)
        logger.info(f'Computed mean: {mean}')
        return mean

    @staticmethod
    def compute_histogram_width(data):
        """Computes the width of the given dataset using a 256 bins histogram.
        This method ignores the zeros. The width is computed as bins.max() - bins.min()

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data

        Returns:
            std (float): non-zero std of the dataset.

        """
        search_data = data[data > 0]
        hist, bins = np.histogram(search_data, bins=256)
        std = bins.max() - bins.min()
        logger.info(f'Computed width: {std}')
        return std

    @staticmethod
    def compute_percentile(data, percentile):
        """Computes the mode of the given dataset using a 100 bins histogram. This method ignores the zeros.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data

        Returns:
            mode (float): non-zero mode of the dataset.

        """
        try:
            non_zero_data = data[np.nonzero(data)]
            return np.percentile(non_zero_data, percentile)
        except:
            return 0

    @staticmethod
    def compute_percentile_median(data, percentile):
        """Computes the mode of the given dataset using a 100 bins histogram. This method ignores the zeros.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data

        Returns:
            mode (float): non-zero mode of the dataset.

        """
        perc = ACRObject.compute_percentile(data, percentile)
        perc_data = data[data >= perc]
        return np.median(perc_data)

    @staticmethod
    def threshold_data(data, intensity, fill=0, greater_than=False):
        """Thresholds the data. Meaning, every pixel with value < intensity or > intensity will be replaced by the
        value in fill. Which side to fill with fill value is driven by the greater_than flag. By default,
        we do the former.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data
            intensity (float): pixel value to use as the threshold
            fill (float, optional): pixel value to use as replacement. Defaults to 0.
            greater_than (bool, optional): Do we fill values that are greater than the specified threshold? Defaults to False.

        Returns:
            data (np.ndarray|np.ma.MaskedArray): data reference.

        """
        mask = data >= intensity if greater_than else data <= intensity
        data[mask] = fill
        return data

    @staticmethod
    def filter_with_dog(data, sigma1=1, sigma2=2, gamma=1.0, iterations=1, ksize=(0, 0)):
        """Performs two Gaussian convolutions with each taking a sigma. Subtracts the second from the first. The idea is
        to eliminate noise.

        Steps:
        ______

            #. Copy the input data
            #. Normalize input data to range [0, 1.0]
            #. Offset results by 0.5 to avoid numpy.power() error message if gamma is not 1.
            #. Apply gamma correction
            #. Obtain 1st Gaussian blurred image using sigma1 for both sigmaX and sigmaY.
            #. Obtain 2nd Gaussian blurred image using sigma2 for both sigmaX and sigmaY.
            #. Subtract => 1 - 2
            #. Remove the offset
            #. Repeat 4 - 8 for n iterations
            #. Restore results intensity range to the input's data type.
            #. Return results

        Notes:
        ______

        Per my testing, this implementation is equivalent to `skimage.filters.difference_of_gaussians` when results are
        binarized. However, there is one fundamental difference and that is that the results there come centered as a
        bellshape which preserve grays. My implementation does not do that. It truly generates pixel subtractions
        for the same input.

        To better mirror the way GIMP implements a difference of Gaussians, I added gamma correction using the
        `Power Law Transform <https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/>`_ on the blurred
        intermediates. The idea is that we can force some of the pixels that are closer to the background closer to 0,
        which can be filtered out elsewhere in your algorithm. Conversely, if you use a larger gamma value, you can
        increase the intensity of pixels and thus bias the output signal towards more surviving pixels if the output
        is meant to be used in a subtraction or threshold operation. Leave the gamma as sis if you simply want
        a DoG without gamma correction.

        An initial offset of 0.5 is applied to the data if gamma != 1.0 to avoid the following error in numpy.power()!

        .. error::

            line 4658, in _lerp
            subtract(b, diff_b_a * (1 - t), out=lerp_interpolation, where=t >= 0.5,
            ValueError: output array is read-only

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data
            sigma1 (float, optional): sigma for the first Gaussian operation. Defaults to 1.
            sigma2 (float, optional): sigma for the second Gaussian operation. This should be bigger than the first value. Defaults to 2.
            gamma (float, optional): value to multiply against each resulting difference to enhance contrast. Defaults to 1.
            iterations (int, optional): How many DoG passes to do. Defaults to 1.

        Returns:
            data (np.ndarray|np.ma.MaskedArray): data reference.

        """
        dtype = data.dtype
        working_data = ACRObject.normalize(data.copy(), max=1, dtype=cv2.CV_32FC1)
        for i in range(iterations):
            working_data = ACRObject.apply_gamma_correction(working_data, gamma)
            blurred = cv2.GaussianBlur(working_data, ksize, sigmaX=sigma1, sigmaY=sigma1)
            blurred2 = cv2.GaussianBlur(blurred, ksize, sigmaX=sigma2, sigmaY=sigma2)
            working_data = cv2.subtract(blurred, blurred2)
        working_data = expand_data_range(working_data, target_type=dtype)
        return working_data

    @staticmethod
    def apply_gamma_correction(data, gamma):
        g = 1 / gamma
        correction = 0.5 if gamma != 1.0 else 0
        return np.power(data + correction, g)

    @staticmethod
    def filter_with_gaussian(data, sigma=1, ksize=(0, 0), dtype=np.uint16):
        noise_removed = cv2.GaussianBlur(data, ksize=ksize, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_ISOLATED)
        return expand_data_range(noise_removed, target_type=dtype)

    @staticmethod
    def normalize(data, max=255, dtype=cv2.CV_8U):
        return cv2.normalize(
            src=data,
            dst=None,
            alpha=0,
            beta=max,
            norm_type=cv2.NORM_MINMAX,
            dtype=dtype,
        )

    @staticmethod
    def resample(data, dx=1, dy=1):
        return cv2.resize(data, dsize=None, fx=dx, fy=dy, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def zoom(data, level=1):
        #target_width = data.shape[0] * level
        #target_height = data.shape[1] * level
        return ACRObject.resample(data, dx=level, dy=level)

    @staticmethod
    def binarize_image(img, percentile=95):
        bin = expand_data_range(img, target_type=np.uint8)
        thr = ACRObject.compute_percentile(bin, percentile)
        logger.info(f'Binarization threshold selected => {thr}')
        bin[bin > thr] = 255
        bin[bin <= thr] = 0
        return bin

    @staticmethod
    def crop_image(img, x, y, width):
        """Return a rectangular subset of a pixel array

        Args:
            img (np.ndarray): dcm.pixelarray
            x (int): x coordinate of centre
            y (int): y coordinate of centre
            width (int): size of the array top subset

        Returns:
            np.ndarray: subset of a pixel array with given width
        """
        crop_x, crop_y = ((
            int(x - width / 2),
            int(x + width / 2)),
                          (
            int(y - width / 2),
            int(y + width / 2),
        ))
        crop_img = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

        return crop_img

    @staticmethod
    def calculate_MTF(erf, dx, dy):
        """Calculate MTF

        Args:
            erf (np.array): array of ?

        Returns:
            tuple: freq, lsf, MTF
        """
        lsf = np.diff(erf)
        N = len(lsf)
        n = (
            np.arange(-N / 2, N / 2)
            if N % 2 == 0
            else np.arange(-(N - 1) / 2, (N + 1) / 2)
        )

        resamp_factor = 8
        Fs = 1 / (
            np.sqrt(np.mean(np.square((dx, dy))))
            * (1 / resamp_factor)
        )
        freq = n * Fs / N
        MTF = np.abs(np.fft.fftshift(np.fft.fft(lsf)))
        MTF = MTF / np.max(MTF)

        zero_freq = np.where(freq == 0)[0][0]
        freq = freq[zero_freq:]
        MTF = MTF[zero_freq:]

        return freq, lsf, MTF

