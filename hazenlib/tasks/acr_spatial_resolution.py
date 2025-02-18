"""
ACR Spatial Resolution (ACR)
____________________________

Reference Materials
+++++++++++++++++++

    *. `Large Phantom Instructions <https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf>`_

Intro
+++++

For this test, resolution in slice 1 of each of the 2 ACR axial series is evaluated. The following procedure is
repeated for each of those series:

    #. Display the slice 1 image.
    #. Magnify the image by a factor of between 2 and 4, keeping the resolution insert visible in the display.
        This is illustrated in Figure 8.
    #. Begin with the leftmost pair of hole arrays, which is the pair with the largest hole size, 1.1 mm.
    #. Look at the rows of holes in the UL array, and adjust the display window and level to best show the holes as
        distinct from one another.
    #. If all 4 holes in any single row are distinguishable from one another, score the image as resolved right-to-left
        at this particular hole size.

To be “distinguishable” or resolved, it is not necessary that image intensity drop to zero between the holes.
To be distinguishable a single window and level setting can be found such that **all 4 holes in at least one row**
are recognizable as points of brighter signal intensity than the spaces between them. Figure 9a shows the typical
appearance of well-resolved holes.

..warning::

        2.5 Causes of Failure and Corrective Actions

        Excessive image filtering can cause failure. Many types of filtering that are used to make the images appear less
        noisy also smooth the image, which blurs small structures. A site that has failed the high-contrast resolution test
        should check that any user selectable image filtering is either turned off, or at least set to the low end of the
        available filter settings.

        Poor eddy current compensation can cause failure. The scanner’s service engineer should check and adjust the eddy
        current compensation if this problem is suspected

Based on the above excerpt from the ACR Guidance Manual, we just need to find one row in which all 4 dots are high
intensity.

Steps
+++++

    #. Create rectangular ROI.
    #. Window Level (which I implemented for the ACR Low Contrast Object Detectability Task).
    #. Threshold.
    #. Identify center to corner pixel.
    #. Build line profile in this row or column.
    #. Find 4 peaks.
    #. Confirm peaks are spaced out.
    #. Confirm that line is valid.
    #. Stop.
    #. Repeat steps for lower box.


Created by Luis M. Santos, M.D.
luis.santos2@nih.gov

02/06/2025
"""

import os
import sys
import traceback
import numpy as np

import cv2
import scipy
import skimage.morphology
import skimage.measure
from matplotlib.image import NonUniformImage

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.logger import logger
from hazenlib.utils import create_rectangular_roi_at, debug_image_sample, wait_on_parallel_results


class ACRSpatialResolution(HazenTask):
    """Spatial resolution measurement class for DICOM images of the ACR phantom

    Inherits from HazenTask class
    """

    ROI_OFFSET = 23         #: 23mm separation between ROIs
    BASE_UL_X_OFFSET = -16  #: -16mm from centroid for 1.1mm resolution array
    BASE_UL_Y_OFFSET = 34   #: 34mm from centroid for 1.1mm resolution array
    BASE_LR_X_OFFSET = -9   #: -9mm from centroid for 1.1mm resolution array
    BASE_LR_Y_OFFSET = 42   #: 42mm from centroid for 1.1mm resolution array

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing spatial resolution measurement
        using slice 1 from the ACR phantom image set

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name,
            input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs,
            optionally path to the generated images for visualisation
        """
        # Identify relevant slices
        dcm = self.ACR_obj.slice_stack[0]

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(dcm)

        try:
            detected_rows = self.get_spatially_resolved_rows(dcm)
            best_resolution = self.get_best_resolution(detected_rows)

            results["measurement"] = {
                "resolution": best_resolution,
                "resolution_units": "mm",
                "rows": {
                    1.1: {
                        'UL': detected_rows[0],
                        'LR': detected_rows[1]
                    },
                    1.0: {
                        'UL': detected_rows[2],
                        'LR': detected_rows[3]
                    },
                    0.9: {
                        'UL': detected_rows[4],
                        'LR': detected_rows[5]
                    }
                }
            }
        except Exception as e:
            print(
                f"Could not calculate the spatial resolution for {self.img_desc(mtf_dcm)} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def write_report(self, dcm, img, center, roi_coords, processed_rois, width):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        #fig, axes = plt.subplots(3, 3)
        fig, axes = plt.subplot_mosaic(
            [
                ['main', 'main', 'main'],
                ['main', 'main', 'main'],
                ['main', 'main', 'main'],
                ['main', 'main', 'main'],
                ['main', 'main', 'main'],
                ['ul1', 'ul2', 'ul3'],
                ['lr1', 'lr2', 'lr3'],
                ['.', '.', '.'],
                ['.', '.', '.']
            ],
            layout="constrained",
            per_subplot_kw={
                'main': {
                    'xbound': (0, 750),
                    'ybound': (0, 500)
                }
            }
        )
        fig.set_size_inches(8, 10)
        #fig.tight_layout(pad=1)

        axes['main'].imshow(img, interpolation="none")
        for i in range(len(roi_coords)):
            roi_center = roi_coords[i]
            rect = patches.Rectangle(
                (roi_center[0] - width / 2, roi_center[1] - width / 2),
                width,
                width,
                linewidth=1,
                edgecolor="w",
                facecolor="none",
            )
            axes['main'].add_patch(rect)
        axes['main'].scatter(center[0], center[1], c="red", s=1)
        axes['main'].axis("off")
        axes['main'].set_title("Centroid + ROI Placement")

        axes['ul1'].set_title("1.1mm")
        ul_roi_center = roi_coords[0]
        axes['ul1'].annotate("UL", (-30, ul_roi_center[1] / 4), xycoords='axes pixels', fontsize='large')
        axes['ul2'].set_title("1.0mm")
        axes['ul3'].set_title("0.9mm")
        lr_roi_center = roi_coords[1]
        axes['lr1'].annotate("LR", (-30, lr_roi_center[1] / 4), xycoords='axes pixels', fontsize='large')

        for i in range(0, len(processed_rois), 2):
            ul = processed_rois[i][1]
            lr = processed_rois[i + 1][1]
            indx = int(i / 2) + 1
            ul_name = f'ul{indx}'
            lr_name = f'lr{indx}'
            axes[ul_name].imshow(ul, interpolation="none")
            axes[ul_name].axis("off")
            axes[ul_name].set_xlabel(ul_name.upper())
            axes[lr_name].imshow(lr, interpolation="none")
            axes[lr_name].axis("off")
            axes[lr_name].set_xlabel(lr_name.upper())

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
        )
        fig.savefig(img_path)
        self.report_files.append(img_path)

    def get_best_resolution(self, detected_rows):
        """Iterates through the detected roi scores. Since these come in UL/LR pairs, the presence of -1 in any of the
        paired items disqualifies the hole array from being a resolved image. Go from 1.1 to 0.9 array and record
        which is the highest resolution we detected. Return this value.

        Args:
            detected_rows (list of int): CList of ints describing each roi's resolved row.

        Returns:
            str: Highest resolution from a set of 1.1, 1.0, and 0.9.
        """
        best_resolution = '1.1'
        for i in range(0, len(detected_rows), 2):
            ul = detected_rows[i]
            lr = detected_rows[i + 1]
            if min(ul, lr) != -1:
                best_resolution = str(np.round(1.1 - 0.1 * int(i / 2), 1))
        return best_resolution

    def get_processed_roi(self, img, width, height, loc):
        """Takes an input image and applies a series of preprocessing steps meant to optimize the output for robust
        detection of signal peaks.

        Preprocessing Steps
        ___________________

            #. Place small ROI using offset from center.
            #. Apply clip windowing method using ROI pixel population.
            #. Crop image at ROI.
            #. Upsample image 3x.
            #. Perform narrow sigma Gaussian denoising.
            #. Perform Difference of Gaussians to drop most of the background and accentuate the main signal.
            #. Binarize image using the 90th percentile as threshold.
            #. Downsample image back to original dimensions. This step simplifies spot detection.
            #. Normalize the image to 1 (makes each pixel Boolean compatible).

        Args:
            img (np.ndarray|np.ma.MaskedArray): Cropped image data
            width (int): Width of rectangular ROI.
            height (int): Height of rectangular ROI.
            loc (tuple of int): Center of where the ROI is expected to be placed.

        Returns:
            tuple of np.ndarray: Denoised windowed output and normalized output.
        """
        roi_x, roi_y = loc

        roi = create_rectangular_roi_at(img, width, height, roi_x, roi_y)
        roi[roi.mask] = 0

        # Windowing step
        non_zero = roi[roi > 0]
        center, window_width = self.ACR_obj.compute_center_and_width(non_zero)
        leveled = self.ACR_obj.apply_window_center_width(roi, center, window_width, function='clip')

        # Cropping step
        crop_img = self.ACR_obj.crop_image(leveled, roi_x, roi_y, width)

        # Upsampling step
        zoom_level = 3  # The ACR prescribes a zoom level of 2 to 4. Thus, a level of 3 can help create an odd grid
        zoom = self.ACR_obj.zoom(crop_img, 3)

        # Denoising step
        smoothed = self.ACR_obj.filter_with_gaussian(zoom, 0.5)

        # Difference of Gaussians step
        dog = self.ACR_obj.filter_with_dog(smoothed, 0.5, 1, gamma=0.5)

        # Binarization step
        binarized = self.ACR_obj.binarize_image(dog, 90)

        # Downsampling step
        downsampled = self.ACR_obj.zoom(binarized, 1 / zoom_level)  # Undo zoom to have nice single scan lines :)

        # Return denoised and normalized (0 to 1) results.
        return smoothed, self.ACR_obj.normalize(downsampled, 1)

    def find_resolved_row(self, roi, ul=True):
        """Goes row by row and detects the number of intensity peaks present.
        To make accuracy robust to signal that bleeds into an otherwise empty row, I check for the max count in pairs of
        rows starting with the first row that contains any signal.

        Detection Steps
        _______________

            #. Rotate ROI 90deg if LR, otherwise use ROI as is.
            #. Iterate row by row and collect the count of signal peaks present.
            #. Trim leading zeros.
            #. From top of signal, iterate pair by pair of rows.
            #. Select the max count of peaks at each row pair.
            #. Trim trailing zeros.
            #. Iterate through each count and stop at the first place with a count of 4.
            #. Report this index location as index + 1. That is the human expected row where the input was deemed resolved.
            #. Otherwise, report -1 to indicate we did not find any resolved rows!


        Args:
            roi (np.ndarray|np.ma.MaskedArray): Cropped image data
            ul (bool): Whether the input is the UL or LR ROI. We want to rotate the LR input such that we can apply the
                        same row traversal

        Returns:
            int: Row number expressing which row was detected as containing 4 peaks. For UL, this number reflect the
                row number starting at the top. For LR, this number reflects the column number starting at the right
                most column and moving to the left (in). Defaults to -1 to demarcate that no rows met the detection
                criteria.
        """
        roi = roi if ul else np.rot90(roi)
        row_length = roi.shape[1]
        vals = np.zeros((roi.shape[0], 1))
        for i in range(len(vals)):
            peaks, intensities = self.ACR_obj.find_n_highest_peaks(roi[i], row_length)
            vals[i] = len(peaks)
        vals = np.trim_zeros(vals, 'f')
        vals = [np.max(vals[i:i + 2]).astype(np.int_) for i in range(0, len(vals), 2)]
        vals = np.trim_zeros(vals, 'b')
        logger.info(f'Row spots detected {vals}')
        for i in range(len(vals)):
            if vals[i] == 4:
                return i + 1
        return -1

    def get_spatially_resolved_rows(self, dcm):
        """Generates a series of ROIs centered around the hole arrays present in the ACR phantom.
        Preprocesses these rois and then attempts to detect hole array.
        These steps are parallelized for maximum processing efficiency.
        Afterwards, write a report if requested. The report has the big picture view on the roi placements and includes
        the preprocessing outputs.

        Args:
            dcm (pydicom.Dataset): DICOM image object

        Returns:
            list of int: Array containing 6 values. One value for each ROI. The format is [ul, lr, ul, lr, ul, lr].
                        Each pair of rois denotes one of three resolution arrays ([1.1, 1.0, 0.9]). Each value
                        represents the row or column in which we detected 4 distinguishable holes.
        """
        img, rescaled, presentation = self.ACR_obj.get_presentation_pixels(dcm)

        # Detect Phantom center.
        dx, dy = float(self.ACR_obj.dx), float(self.ACR_obj.dy)
        cxy, _ = self.ACR_obj.find_phantom_center(img, dx, dy)
        width = int(np.round(12 / dx))

        logger.info(f"Center           => {cxy}")
        logger.info(f"Pixel Resolution => {dx, dy}")
        logger.info(f"ROI width        => {width}")

        # TODO: Add Unit tests
        # TODO: Validate results

        # Generate preprocessed ROIs
        task_args = []
        roi_coords = []
        for i in range(3):
            # Request UL ROI
            x_off = (self.BASE_UL_X_OFFSET / dx) + i * self.ROI_OFFSET / dx
            ul_x, ul_y = (int(cxy[0] + x_off), int(cxy[1] + self.BASE_UL_Y_OFFSET / dy))
            logger.info(f"UL ROI Center for Array {i} => {ul_x, ul_y}")
            task_args.append((rescaled, width, width, (ul_x, ul_y)))
            roi_coords.append((ul_x, ul_y))

            # Request LR ROI
            x_off = (self.BASE_LR_X_OFFSET / dx) + i * self.ROI_OFFSET / dx
            lr_x, lr_y = (int(cxy[0] + x_off), int(cxy[1] + self.BASE_LR_Y_OFFSET / dy))
            logger.info(f"LR ROI Center for Array {i} => {lr_x, lr_y}")
            task_args.append((rescaled, width, width, (lr_x, lr_y)))
            roi_coords.append((lr_x, lr_y))
        processed_rois = wait_on_parallel_results(self.get_processed_roi, task_args)

        # Detect resolved row/column in each ROI.
        task_args.clear()
        for i in range(0, len(processed_rois), 2):
            task_args.append((processed_rois[i][1], True))
            task_args.append((processed_rois[i + 1][1], False))
        detected_rows = wait_on_parallel_results(self.find_resolved_row, task_args)

        logger.info(f'Detected rows => {detected_rows}')

        if self.report:
            self.write_report(dcm, img, cxy, roi_coords, processed_rois, width)

        return detected_rows
