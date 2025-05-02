"""
ACR Slice Thickness
___________________

Reference
_________

`ACR Large Phantom Guidance PDF <https://accreditationsupport.acr.org/helpdesk/attachments/11093487417>`_

Intro
_____

The slice thickness accuracy test assesses the accuracy with which a slice of specified thickness is achieved.
The prescribed slice thickness is compared with the measured slice thickness.

The ramps appear in a structure called the slice thickness insert. Figure 10 shows an image of slice 1 with
the slice thickness insert and signal ramps identified. The two ramps are crossed: one has a negative slope
and the other a positive slope with respect to the plane of slice 1. They are produced by cutting 1 mm wide
slots in a block of plastic. The slots are open to the interior of the phantom and are filled with the same
solution that fills the bulk of the phantom.

The signal ramps have a slope of 10 to 1 with respect to the plane of slice 1, which is an angle of about 5.71°.
Therefore, the signal ramps will appear in the image of slice 1 with a length that is 10 times the thickness of
the slice. If the phantom is misaligned from right-left, one ramp will appear longer than the other. The
crossed ramps allow for correction of the error introduced by right-left misalignment, and the slice thickness
formula takes that into account.

ACR Guidelines
______________

ACR Algorithm
+++++++++++++

    #. Display slice 1, and magnify the image by a factor of 2 to 4, keeping the slice thickness insert fully
        visible on the screen.
    #. Adjust the display level so that the signal ramps are well visualized.
        *. The ramp signal is much lower than that of surrounding water, so usually it will be necessary
            to lower the display level substantially and narrow the window.
    #. Place a rectangular ROI at the middle of each ramp as shown below in Figure 11.
        *. Note the mean signal values for each of these two ROIs and then average those values.
        *. The result is a number approximating the mean signal in the middle of the ramps.
        *. An elliptical ROI may be used if a rectangular one is unavailable.
    #. Lower the display level to half of the average ramp signal calculated in step 3.
        *. Leave the display window set to its minimum.
    #. Use the on-screen distance measurement tool to measure the lengths of the top and bottom ramps.
        This is illustrated below in Figure 12. Record these lengths and compare to the action limits.

Our Approximation
+++++++++++++++++

    #. Find the phantom center.
    #. Zoom input x4.
    #. Crop insert around initial center.
    #. Apply Window Width and Window Level based on cropped region.
    #. Identify the Y coordinates by sampling a line profile at the sample crop center and finding highest 2 peaks.
    #. Identify the X coordinates by sampling line profiles in the horizontal direction going through the Y points.
    #. Place rectangular ROIs at those centers with a fixed width.
    #. Apply WL based on ROI averages.
    #. Identify widths.
    #. Use ACR formula for results.

ACR Scoring Rubric
++++++++++++++++++

The slice thickness is calculated using the following formula:

    Slice thickness = 0.2 x (top x bottom)/(top + bottom)

In the formula above, the `top` and `bottom` are the measured lengths of the top and bottom signal ramps.

**Note:** 0.2 is a unitless factor that corrects for rotation of the phantom about the vertical (y) axis.

For example, if the top signal ramp were 59.5mm long and the bottom ramp were 47.2mm long, then the
calculated slice thickness would be:

    Slice thickness = 0.2 x (59.5mm x 47.2mm)/(59.5mm + 47.2mm) = 5.26 mm.


Notes
_____

..note::

    A failure of this test means that the scanner is producing slices of substantially different thickness from the
    prescribed thickness. This problem will generally not occur in isolation since the scanner deficiencies that
    can cause it may also cause other image problems. Therefore, the implications of a failure are not just that
    the slices are too thick or thin, but can also result in poor image contrast and low SNR.

..warning::

    When making these measurements, **be careful to fully cover the widths of the ramp with the
    ROIs** in the top-bottom direction, but not to allow the ROIs to stray outside the ramps into adjacent
    high- or low-signal regions. If there is a large difference,(that is, more than 20%), between the signal
    values obtained for the ROIs, it is often due to one or both of the ROIs including regions outside the
    ramps.

Documented by Luis M. Santos, M.D.
luis.santos2@nih.gov


"""

import os
import sys
import traceback
import numpy as np

import scipy
import skimage.morphology
import skimage.measure
from scipy.signal import peak_widths

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.logger import logger
from hazenlib.utils import get_image_orientation, debug_image_sample


class ACRSliceThickness(HazenTask):
    """Slice width measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)
        self.SAMPLING_LINE_WIDTH = 4 / self.ACR_obj.dx   # How many pixel lines to use in the sampling during ramp line profiling.
        self.RAMP_HEIGHT = 4.5 / self.ACR_obj.dx         # I measured the ramp height to be about 5mm on PACS, but testing shows it might be slightly less??
        self.RAMP_Y_OFFSET = 1 / self.ACR_obj.dx         # 1mm adjustment off center to grab the bottom ramp. There's technically a 2mm gap between slots.
        self.INSERT_ROI_HEIGHT = 10 / self.ACR_obj.dx    # Allow just enough space for slots but exclude insert boundaries
        self.INSERT_ROI_WIDTH = 150 / self.ACR_obj.dx    # Allow enough space to capture the slots which might be R-L offsetted.
        self.CROPPED_ROI_WIDTH = 150 / self.ACR_obj.dx   # Allow enough space to capture the slots which might be R-L offsetted.
        self.CROPPED_ROI_HEIGHT = 20 / self.ACR_obj.dx   # Capture slots plus some surrounding areas to help visualization in report.
        self.WINDOW_ROI_WIDTH = 10 / self.ACR_obj.dx     # Rectangle that captures enough of a population at the center to determine proper mean signal of slots.
        self.WINDOW_ROI_HEIGHT = 5 / self.ACR_obj.dx     # Rectangle that captures enough of a population at the center to determine proper mean signal of slots.

    def run(self) -> dict:
        """Main function for performing slice width measurement
        using slice 1 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation.
        """
        # Identify relevant slice
        slice_thickness_dcm = self.ACR_obj.slice_stack[0]
        # TODO image may be 90 degrees cw or acw, could use code to identify which or could be added as extra arg

        ori = get_image_orientation(slice_thickness_dcm)
        if ori == 'Sagittal':
            # Get the pixel array from the DICOM file
            img = slice_thickness_dcm.pixel_array

            # Rotate the image 90 degrees clockwise

            rotated_img = np.rot90(img, k=-1)  # k=-1 for 90 degrees clockwise

            # Update the pixel array in the DICOM object
            slice_thickness_dcm.PixelData = rotated_img.tobytes()

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(slice_thickness_dcm)

        try:
            thickness_results = self.get_slice_thickness(slice_thickness_dcm)
            results["measurement"] = {"slice width mm": round(thickness_results['thickness'], 2)}
            results["ramps"] = thickness_results["ramps"]
        except Exception as e:
            logger.error(
                f"Could not calculate the slice thickness for {self.img_desc(slice_thickness_dcm)} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def write_report(self, dcm, center, results):
        import matplotlib.pyplot as plt

        (center_x, center_y) = center
        fig, axes = plt.subplot_mosaic(
            [
                ['.', '.', '.', '.', '.', '.'],
                ['.', 'main', 'main', 'main', 'main', '.'],
                ['.', 'main', 'main', 'main', 'main', '.'],
                ['.', 'main', 'main', 'main', 'main', '.'],
                ['.', 'main', 'main', 'main', 'main', '.'],
                ['.', 'main', 'main', 'main', 'main', '.'],
                ['.', 'main', 'main', 'main', 'main', '.'],
                ['.', 'main', 'main', 'main', 'main', '.'],
                ['.', '.', '.', '.', '.', '.'],
                ['.','insert', 'insert', 'insert', 'insert', '.'],
                ['.','insert', 'insert', 'insert', 'insert', '.'],
                ['.','top', 'top', 'top', 'top', '.'],
                ['.','top', 'top', 'top', 'top', '.'],
                ['.','bottom', 'bottom', 'bottom', 'bottom', '.'],
                ['.','bottom', 'bottom', 'bottom', 'bottom', '.'],
            ],
            layout="constrained",
            per_subplot_kw={
                'main': {
                    'xbound': (0, 750),
                    'ybound': (0, 500)
                }
            }
        )
        fig.set_size_inches(8, 8)
        #fig.tight_layout(pad=4)

        # Centroid
        axes['main'].imshow(results['img'], cmap='viridis')
        axes['main'].scatter(center_x, center_y, c="red")
        axes['main'].axis("off")
        axes['main'].set_title("Window Leveled + Centroid Location")

        # Centroid
        axes['insert'].imshow(results['rois']['insert'], cmap='viridis')
        axes['insert'].axis("off")
        axes['insert'].set_title("Insert")

        # Top Ramp
        top_center = results['ramps']['top']['center']
        top_width = results['ramps']['top']['width']
        top_half_width = top_width / 2
        axes['top'].imshow(results['rois']['top'], cmap='viridis')
        axes['top'].scatter(*top_center, c="blue")
        axes['top'].plot(
            [top_center[0] - top_half_width, top_center[0] + top_half_width],
            [top_center[1], top_center[1]],
            "b-"
        )
        axes['top'].axis("off")
        axes['top'].set_title("Top Ramp")

        # Bottom Ramp
        bottom_center = results['ramps']['bottom']['center']
        bottom_width = results['ramps']['bottom']['width']
        bottom_half_width = bottom_width / 2
        axes['bottom'].imshow(results['rois']['bottom'], cmap='viridis')
        axes['bottom'].scatter(*bottom_center, c="red")
        axes['bottom'].plot(
            [bottom_center[0] - bottom_half_width, bottom_center[0] + bottom_half_width],
            [bottom_center[1], bottom_center[1]],
            "r-"
        )
        axes['bottom'].axis("off")
        axes['bottom'].set_title("Bottom Ramp")

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}_slice_thickness.png")
        )
        fig.savefig(img_path)
        self.report_files.append(img_path)
        return img_path

    def find_insert(self, img, centre):
        """
        This is a very important step. Proper isolation of the insert makes ramp detection and length calculation
        much easier!

        Steps
        +++++

            #. Crops the input image into a large cropped region.
            #. Crops the input image into a small rectangular region.
            #. Use the small rectangular window to determine the window level.
            #. Use the window level to determine the half level per ACR guidelines. Technically, I am short-cicuiting
                this portion of the guidelines, but we gain a robust way to extract the ramps.
            #. Level the large ROI.
            #. Test large ROI for relative Y center coordinate.
            #. Use large ROI relative center to determine Insert ROI window and crop it.
            #. Return large ROI and Insert ROI. Both are properly windowed.

        Args:
            img (np.ndarray): input image.
            centre (tuple[float|int]): center of the input image.

        Returns:
            tuple[np.ndarray]: Large crop window and insert ROI. Both are rescaled to the half window level.
        """
        # Create a crop window of the general insert region. This includes portions of the bordering bright pixels
        # with ample space to ensure capture of the insert regardless of centroid errors.
        interest_region = self.ACR_obj.crop_image(img, centre[0],
                                                centre[1],
                                                self.CROPPED_ROI_WIDTH,
                                                self.CROPPED_ROI_HEIGHT)
        # Grab a crop of a rectangle around the centroid.
        # This is meant to ensure we get the correct half level in an approximation of ACR guidelines.
        window_region = self.ACR_obj.crop_image(img, centre[0],
                                                centre[1],
                                                self.WINDOW_ROI_WIDTH,
                                                self.WINDOW_ROI_HEIGHT)

        level, width = self.ACR_obj.compute_center_and_width(window_region)
        half_level = level / 2

        # Generate leveled version of the general ROI. This is effectively a binarization step. It is very important
        # of a step since it simplifies centering improvements and edge/peak detection.
        leveled_interest_region = self.ACR_obj.apply_window_center_width(interest_region, half_level, 0)

        # Find out the relative center of the Insert ROI with respect to the general region of interest that includes
        # the bordering bright pixels.
        # Basically, we do a line profile to determine the true center of the slot region.
        # This helps with correcting any errors or biases in centering introduced by the centroid detection.
        center_y = int(self.find_insert_region_center_y(self.ACR_obj.invert_image(leveled_interest_region)))
        center_x = int(np.ceil(leveled_interest_region.shape[1] / 2))
        logger.info(f'Relative Center of Insert ROI is => ({center_x}, {center_y})')

        # The grand finale, crop a perfect insert rectangle that contains the slots
        insert_region = self.ACR_obj.crop_image(leveled_interest_region, center_x,
                                                int(center_y),
                                                self.INSERT_ROI_WIDTH,
                                                self.INSERT_ROI_HEIGHT)
        return leveled_interest_region, insert_region

    def find_insert_region_center_y(self, insert_region):
        # the line profile skips first pixel.
        profile = skimage.measure.profile_line(
            insert_region,
            (-1, 0),
            (insert_region.shape[0], 0),
            mode="constant"
        )
        x_diff = np.diff(profile)
        abs_x_diff_profile = np.absolute(x_diff)

        peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_x_diff_profile, 5)
        return np.ceil((peaks[0] + peaks[-1]) / 2)

    def find_ramp_regions(self, insert):
        """
        Crop the top and bottom ramp relative to the center of the insert.

        Args:
            insert (np.ndarray): ramp to use in calculation of width.

        Returns:
            tuple: top slot ROI, bottom slot ROI
        """
        center_y = insert.shape[0] / 2
        top_roi_sample = self.ACR_obj.crop_image(insert, 0, abs(center_y - self.RAMP_HEIGHT + self.RAMP_Y_OFFSET), insert.shape[1], self.RAMP_HEIGHT, mode=None)
        bottom_roi_sample = self.ACR_obj.crop_image(insert, 0, abs(center_y + self.RAMP_Y_OFFSET), insert.shape[1], self.RAMP_HEIGHT, mode=None)
        return top_roi_sample, bottom_roi_sample

    def find_ramp_center_width(self, ramp):
        """Given a ramp roi, perform a horizontal line profile looking for mean values.
        Then do a rolling difference to accentuate the peaks.
        Finally, collect the peaks.

        Use the peaks to identify the edges of the ramp.
        Return the difference as this is the relative difference of the ramp.

        We also return a center point calculated relative to the given ramp for presentation purposes.

        ..note::

            I do a line profile using 4 lines from the input ramp. These 4 lines get collapsed into a 1-D array via
            the Numpy function mean. Essentially, the line profile is the mean line through the slot.

        Args:
            ramp (np.ndarray): ramp to use in calculation of width.

        Returns:
            tuple: center point of ramp (x, y), width.
        """
        profile = skimage.measure.profile_line(
            ramp,
            (0, 0),
            (0, ramp.shape[1]),
            linewidth=int(np.round(self.SAMPLING_LINE_WIDTH)),
            reduce_func=np.mean,
            mode="constant"
        )
        x_diff = np.diff(profile)
        abs_x_diff_profile = np.absolute(x_diff)

        peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_x_diff_profile, 5)
        left_point = np.min(peaks)
        right_point = np.max(peaks)
        # Up until this point, the width was based pixels per mm, here we adjust it back to mm
        length = (right_point - left_point) * self.ACR_obj.dx
        cx = np.round(float(left_point + length / 2))
        return (cx, np.round(float(ramp.shape[0] / 2))), float(length)

    def find_ramps(self, img, centre):
        """
        Follows these simple steps:

            #. Obtain insert ROI.
            #. Obtain the ramp ROIs from the insert ROI.
            #. Compute top ROI center and width.
            #. Compute bottom ROI center and width.

        Args:
            img (np.ndarray): input image.
            centre (tuple[float|int]): center of the input image.

        Returns:
            dict: Structure containing the cropped rois and the ramp center and width.
        """
        general_roi, insert_roi = self.find_insert(img, centre)

        top_roi_leveled, bottom_roi_leveled = self.find_ramp_regions(insert_roi)

        top_c, top_width = self.find_ramp_center_width(top_roi_leveled)

        bottom_c, bottom_width = self.find_ramp_center_width(bottom_roi_leveled)

        return {
            'rois': {
                'insert': general_roi,
                'top': top_roi_leveled,
                'bottom': bottom_roi_leveled
            },
            'ramps': {
                'top': {
                    'center': top_c,
                    'width': top_width,
                },
                'bottom': {
                    'center': bottom_c,
                    'width': bottom_width,
                }
            }
        }

    def compute_thickness(self, top_width, bottom_width):
        """ Given the top ramp's width and the bottom ramp's width, compute the slice thickness in units of mm.

        Formula
        _______

            Slice thickness = 0.2 x (top x bottom)/(top + bottom)

        ..note::

            Per the ACR, "0.2 is a unitless factor that corrects for rotation of the phantom about the vertical (y) axis."
            We use such factor in the formula.

        Args:
            top_width (float): Top ramp's calculated width.
            bottom_width (float): Bottom ramp's calculated width.

        Returns:
            float: slice thickness in mm.
        """
        return 0.2 * (top_width * bottom_width) / (top_width + bottom_width)

    def get_slice_thickness(self, dcm):
        """Measure slice thickness and report it.



        Args:
            dcm (pydicom.Dataset): DICOM image object.

        Returns:
            dict: Dictionary of the form.
                {
                    "rois": <dict of np.ndarray>,
                    "ramps": <dict>,
                    "thickness": <float>,
                    "img": <np.ndarray>
                }
        """
        img, rescaled, presentation = self.ACR_obj.get_presentation_pixels(dcm)
        cxy, _ = self.ACR_obj.find_phantom_center(rescaled, self.ACR_obj.dx, self.ACR_obj.dy)

        ramps = self.find_ramps(rescaled, cxy)

        # Per the ACR formula. Slice thickness = 0.2 x (top x bottom)/(top + bottom)
        top_width = ramps['ramps']['top']['width']
        bottom_width = ramps['ramps']['top']['width']
        thickness = self.compute_thickness(top_width, bottom_width)

        results = {
            **ramps,
            'thickness': thickness,
            'img': rescaled,
        }

        if self.report:
            self.write_report(dcm, cxy, results)

        return results
