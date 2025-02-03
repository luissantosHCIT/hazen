"""
ACR Low-Contrast Object Detectability Task
__________________________________________

Reference
_________

`ACR Large Phantom Guidance PDF <https://accreditationsupport.acr.org/helpdesk/attachments/11093487417>`_

Intro
_____

Performs a series of Image Processing steps that attempts to

    #. Isolate the signal of the spots inside the circle ROI in slices 8 through 11.
    #. Predict where spots should be present in ROI.
    #. Walk through the predicted spot ROIs.
    #. Asses if spoke is valid.
    #. Stop at first invalid spoke.
    #. Calculate the slice score.

A lot of the tuning performed attempts to strike a balance between the real number of high intensity signals and
avoiding overestimation. As a result, my current attempt is biased towards underestimation if the signal is too
small to survive thresholding.

ACR Guidelines
______________

ACR Algorithm
+++++++++++++

    #. Display slice 11, which has the highest contrast objects. Adjust the display window width and level
        settings for best visibility of the low contrast objects. This will usually require a fairly narrow window
        width and careful adjustment of the level to best distinguish the objects from the background.
    #. Count the number of complete spokes. Begin counting with the spoke having the largest diameter
        disks; this spoke is at 12 o’clock or slightly to the right of 12 o’clock, and is referred to as spoke 1.
        Count clockwise from spoke 1 until a spoke is reached where one or more of the disks is not
        discernible from the background. A spoke is complete only if all three disks are discernible. Count
        complete spokes, not individual disks.
    #. The score for this slice is the number of complete spokes. Record the score.
    #. Repeat the procedure to determine the number of visible spokes for the remaining LCD images.

ACR Scoring Rubric
++++++++++++++++++

For each series, record the number of complete spokes visible on each slice, then sum the values for all four
slices to determine the total LCD score. For example, if the ACR T2 series scored 3 spokes in slice 8, 5 spokes
in slice 9, 9 spokes in slice 10, and 10 spokes in slice 11; the total score for the ACR T2 series would be 3 + 5
+ 9 + 10 = 27.

Nominal Field  ACR T1 LCD       ACR T2 LCD
Strength       Limit           Limit
               (total spokes)  (total spokes)
_____________  _______________ ______________
<1.5T          ≥7              ≥7
1.5T - <3T     ≥30             ≥25
3T             ≥37             ≥37

Created by Luis M. Santos, M.D.
luis.santos2@nih.gov

13/01/2022
"""

import sys
import os
import traceback

import cv2
import numpy as np
from matplotlib.pyplot import subplots as plt_subplots

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib import logger
from hazenlib.utils import compute_radius_from_area, create_circular_roi_at, expand_data_range, wait_on_parallel_results


class ACRObjectDetectability(HazenTask):
    """Uniformity measurement class for DICOM images of the ACR phantom."""

    DOT_SEPARATION = 12.8               #: 12 to 14 mm from center to center of each circle.
    DOT_ANGLE = np.deg2rad(36)          #: Each spoke is at this angle of separation.
    SLICE_ANGLE_OFFSET = np.deg2rad(9)  #: Each subsequent slice
    START_ANGLE = np.deg2rad(90)        #: Slice 0 has spots at a 90 deg angle
    ORIG_SPOKE_RADII = {
        0: 3.5,
        1: 3.1945,
        2: 2.889,
        3: 2.5835,
        4: 2.278,
        5: 1.9725,
        6: 1.667,
        7: 1.3615,
        8: 1.056,
        9: 0.75,
    }
    SPOKE_RADII = {                     #: Radius used for each spoke spot. Meaning, in spoke 0 all spots have the same
        0: 2.0,                         #: radius.
        1: 2.0,
        2: 2.0,
        3: 2.0,
        4: 2.0,
        5: 2.0,
        6: 2.0,
        7: 2.0,
        8: 2.0,
        9: 2.0,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)
        # Reference slice
        img = self.ACR_obj.slice_stack[6].pixel_array
        # Calculate center
        self.center, _ = self.ACR_obj.find_phantom_center(
            img, self.ACR_obj.dx, self.ACR_obj.dy
        )
        logger.info(f'Phantom centroid set to {self.center}')
        # Find inner center y coordinates which is slightly offsetted
        offsetted_y = np.round(self.center[1] + 7 / self.ACR_obj.dy)
        self.center = (self.center[0] - self.ACR_obj.dx, offsetted_y)
        logger.info(f'Inner ROI centroid set to {self.center}')
        # Required pixel radius to produce ~75cm2 ROI
        self.r_inner = compute_radius_from_area(75, self.ACR_obj.dx)
        # Required pixel radius to produce ~15 ROI
        self.r_noise = compute_radius_from_area(15, self.ACR_obj.dx)
        # Required pixel radius to produce ~0.25 ROI
        self.r_spot = compute_radius_from_area(0.25, self.ACR_obj.dx)

    def run(self) -> dict:
        """Main function for performing uniformity measurement using slice 7 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM
                Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path
                to the generated images for visualisation
        """
        slices = [
            self.ACR_obj.slice_stack[7],
            self.ACR_obj.slice_stack[8],
            self.ACR_obj.slice_stack[9],
            self.ACR_obj.slice_stack[10],
        ]
        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = [self.img_desc(sl) for sl in slices]

        try:
            r = self.get_spokes_and_scores(slices)
            results.update(r["meta"])
        except Exception as e:
            logger.error(
                f"Could not calculate the number of spokes for the Low Contrast Object Detectability Task "
                f"because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def write_report(self, slices, center, results):
        data = results["data"]

        arg_list = [(slices[i], data[i], center) for i in range(len(slices))]
        self.report_files = wait_on_parallel_results(self.write_report_slice, arg_list)

    def write_report_slice(self, dcm, img_result, center, theta=np.linspace(0, 2 * np.pi, 360)):
        (center_x, center_y) = center
        fig, axes = plt_subplots(3, 1)
        fig.set_size_inches(8, 16)
        fig.tight_layout(pad=4)

        # Centroid
        axes[0].imshow(img_result[-1]['img'][0], cmap='gray', vmin=0, vmax=255)
        axes[0].scatter(center_x, center_y, c="red")
        axes[0].axis("off")
        axes[0].set_title("Window Leveled + Centroid Location")

        # DoG
        axes[1].imshow(img_result[-1]['img'][1])
        axes[1].axis("off")
        axes[1].set_title("Difference of Gaussians")

        axes[2].imshow(img_result[-1]['img'][2], cmap='gray', vmin=0, vmax=255)
        spokes = len(img_result) - 1
        for j in range(spokes):
            spot_center1, spot_center2, spot_center3 = img_result[j]['centers']
            axes[2].plot(
                self.r_spot * np.cos(theta) + spot_center1[0],
                self.r_spot * np.sin(theta) + spot_center1[1],
                c="green",
            )
            axes[2].plot(
                self.r_spot * np.cos(theta) + spot_center2[0],
                self.r_spot * np.sin(theta) + spot_center2[1],
                c="green",
            )
            axes[2].plot(
                self.r_spot * np.cos(theta) + spot_center3[0],
                self.r_spot * np.sin(theta) + spot_center3[1],
                c="green",
            )
        axes[2].axis("off")
        axes[2].set_title("Valid Spokes (dilated)")

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
        )
        fig.savefig(img_path)
        return img_path

    @staticmethod
    def calculate_spot_location(center, angle, spot_separation, spot):
        x_dist, y_dist = np.floor(np.cos(angle) * spot_separation * spot), np.floor(
            np.sin(angle) * spot_separation * spot)
        return (center[0] + x_dist, center[1] - y_dist)

    def detect_spot(self, img, center, angle, spot_separation, spot_radius, spot):
        x, y = self.calculate_spot_location(center, angle, spot_separation, spot)
        return create_circular_roi_at(img, spot_radius, x, y), (x, y)

    def detect_spoke(self, img, center, slice_num, spoke):
        spot_radius = int(np.ceil(self.SPOKE_RADII[spoke] / self.ACR_obj.dx))
        spot_separation = self.DOT_SEPARATION / self.ACR_obj.dx
        angle = self.START_ANGLE - (spoke * self.DOT_ANGLE) - (self.SLICE_ANGLE_OFFSET * slice_num)

        # Generate individual spot masks on image.
        spot1 = self.detect_spot(img, center, angle, spot_separation, spot_radius, 1)
        spot2 = self.detect_spot(img, center, angle, spot_separation, spot_radius, 2)
        spot3 = self.detect_spot(img, center, angle, spot_separation, spot_radius, 3)

        return spot1, spot2, spot3

    @staticmethod
    def combine_masks(spots, target_mask):
        # Combine the spot masks into a master spoke mask
        spot1, spot2, spot3 = spots
        combined_mask = np.ma.mask_or(~spot1[0].mask, np.ma.mask_or(~spot2[0].mask, ~spot3[0].mask))
        return np.ma.mask_or(combined_mask, target_mask)

    @staticmethod
    def binarize(img):
        bin = expand_data_range(img, target_type=np.uint8)
        thr = np.percentile(bin[np.nonzero(bin)], 98.7)
        logger.info(f'Binarization threshold selected => {thr}')
        bin[bin > thr] = 255
        bin[bin <= thr] = 0
        return bin

    def compute_score(self, feature_data, center, slice_num):
        spoke_results = {}
        combined_mask = feature_data.mask if isinstance(feature_data, np.ma.MaskedArray) else feature_data > 0
        for spoke in range(10):
            spot1, spot2, spot3 = self.detect_spoke(feature_data, center, slice_num, spoke)
            spot_score = sum([spot1[0].sum() > 0, spot2[0].sum() > 0, spot3[0].sum() > 0])
            valid = spot_score == 3
            if valid:
                combined_mask = self.combine_masks((spot1, spot2, spot3), combined_mask)
                spoke_results[spoke] = {
                    "spots": spot_score,
                    "centers": [spot1[1], spot2[1], spot3[1]],
                }
            else:
                break
        spoke_results[-1] = {
            "mask": combined_mask
        }
        return spoke_results

    def detect_objects(self, img, center_x, center_y, slice_num):
        logger.info(f'Processing slice # {8 + slice_num}')

        # First, let do a light Gaussian pass to help remove some of the crazy noise and improve SNR.
        noise_removed = self.ACR_obj.filter_with_gaussian(img)

        # Now, we can ready the ROI on which to focus on extracting the signal
        inner_roi = create_circular_roi_at(noise_removed, self.r_inner, center_x, center_y)
        inner_roi[inner_roi.mask] = 0

        # Find noise sampling ROI
        noise_roi = create_circular_roi_at(inner_roi, self.r_noise, center_x, center_y)
        noise_roi[noise_roi.mask] = 0

        # Compute the window level and width in the noise sampling area.
        center, width = self.ACR_obj.compute_width_and_center(noise_roi)
        logger.info(f'Target Windowing => {center}, {width}')

        # Apply the previously computed window settings to the inner ROI window.
        contrasted = self.ACR_obj.apply_window_width_center(inner_roi, center, width * 1 / self.ACR_obj.dx)

        # Perform Difference of Gaussians to further isolate relevant pixels
        # Using a large gamma to allow high intensities to survive the DoG operation.
        # Gamma correction has a profound effect on remaining signal just like the selection of sigma2.
        # Gamma correction here helps with fine-tuning to approximate what I thought is reality for the GE dataset which
        # was noisier than more ideal scans. GE => gamma = 20, sigma2 = 3.5
        dog = self.ACR_obj.filter_with_dog(contrasted, 1 / self.ACR_obj.dx, 2 / self.ACR_obj.dx, gamma=100)
        dog = np.ma.masked_array(dog, mask=inner_roi.mask, fill_value=0)

        # Binarize the results
        binarized = self.binarize(dog)

        # Dilate the signal that is present.
        dilated = cv2.dilate(binarized, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        dilated = np.ma.masked_array(dilated, mask=inner_roi.mask, fill_value=0)

        # Count spots and spokes clockwise
        # A spot is valid if its max intensity is above relative threshold
        # A spoke is valid if it contains 3 successive spots in diagonal.
        results = self.compute_score(dilated, (center_x, center_y), slice_num)

        windowed = expand_data_range(contrasted)
        windowed[inner_roi.mask] = 0
        results[-1]['img'] = [windowed, dog, dilated]

        return results

    def get_spokes_and_scores(self, slices):
        """Calculates the percent integral uniformity (PIU) of a DICOM pixel array. \n
        Iterates with a ~1 cm^2 ROI through a ~200 cm^2 ROI inside the phantom region,
        and calculates the mean non-zero pixel value inside each ~1 cm^2 ROI. \n
        The PIU is defined as: `PIU = 100 * (1 - (max - min) / (max + min))`, where \n
        'max' and 'min' represent the maximum and minimum of the mean non-zero pixel values of each ~1 cm^2 ROI.

        Args:
            dcm (pydicom.Dataset): DICOM image object to calculate uniformity from.

        Returns:
            float: value of integral uniformity.
        """
        (center_x, center_y) = self.center

        results = {
            "meta": {},
            "data": {}
        }

        # Run processing jobs
        jobs = [(slices[i].pixel_array, center_x, center_y, i) for i in range(4)]
        result_data = wait_on_parallel_results(self.detect_objects, jobs)

        # Collect data and final score
        score = 0
        for i in range(4):
            results["data"][i] = result_data[i]
            score += len(results["data"][i]) - 1

        # Append meta data about results
        results["meta"]["field_strength"] = slices[-1].MagneticFieldStrength
        results["meta"]["score"] = score
        logger.info(results["meta"])

        # Generate report
        if self.report:
            logger.info('Writing report ... ')
            self.write_report(slices, (center_x, center_y), results)

        return 0
