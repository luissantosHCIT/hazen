"""
ACR Spatial Resolution (ACR)
____________________________

Reference Materials
+++++++++++++++++++

    *. `Large Phantom Instructions <https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf>`_
    *. `MTF-50 Paper <https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-5-6040&id=281325>`_

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

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.logger import logger
from hazenlib.utils import create_rectangular_roi_at, debug_image_sample


class ACRSpatialResolution(HazenTask):
    """Spatial resolution measurement class for DICOM images of the ACR phantom

    Inherits from HazenTask class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing spatial resolution measurement
        using slice 1 from the ACR phantom image set

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """
        # Identify relevant slices
        mtf_dcm = self.ACR_obj.slice_stack[0]

        rot_ang = self.ACR_obj.determine_rotation(mtf_dcm.pixel_array)
        if np.abs(rot_ang) < 3:
            logger.warning(
                f"The estimated rotation angle of the ACR phantom is {np.round(rot_ang, 3)} degrees, which "
                f"is less than the recommended 3 degrees. Results will be unreliable!"
            )

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(mtf_dcm)

        try:
            raw_res, fitted_res = self.get_mtf50(mtf_dcm)
            results["measurement"] = {
                "estimated rotation angle": round(rot_ang, 2),
                "raw mtf50": round(raw_res, 2),
                "fitted mtf50": round(fitted_res, 2),
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

    def write_report(self, dcm, img, ramp_x, ramp_y, width):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, axes = plt.subplots(2, 1)
        fig.set_size_inches(8, 10)
        fig.tight_layout(pad=4)

        axes[0].imshow(img, interpolation="none")
        rect = patches.Rectangle(
            (ramp_x - width / 2, ramp_y - width / 2),
            width,
            width,
            linewidth=1,
            edgecolor="w",
            facecolor="none",
        )
        axes[0].add_patch(rect)
        axes[0].scatter(ramp_x, ramp_y, c="red", s=1)
        axes[0].axis("off")
        axes[0].set_title("Segmented Edge")

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
        )
        fig.savefig(img_path)
        self.report_files.append(img_path)

    def y_position_for_ramp(self, img, cxy):
        """Identify the y coordinate of the ramp

        Args:
            img (np.ndarray): dcm.pixelarray
            cxy (tuple): x,y coordinates of the object centre

        Returns:
            float: y coordinate of the ramp min
        """
        investigate_region = int(np.ceil(5.5 / self.ACR_obj.dy).item())

        if np.mod(investigate_region, 2) == 0:
            investigate_region = investigate_region + 1

        line_profile_y = skimage.measure.profile_line(
            img,
            (cxy[1] - 2 * investigate_region, cxy[0]),
            (cxy[1] + 2 * investigate_region, cxy[1]),
            mode="constant",
        ).flatten()

        abs_diff_y_profile = np.absolute(np.diff(line_profile_y))
        y_peaks = scipy.signal.find_peaks(abs_diff_y_profile, height=1)
        pk_heights = y_peaks[1]["peak_heights"]
        pk_ind = y_peaks[0]
        highest_y_peaks = pk_ind[(-pk_heights).argsort()[:2]]
        y_locs = highest_y_peaks - 1

        height_pts = cxy[1] - 2 * investigate_region - 1 + y_locs

        y = np.min(height_pts) + 2

        return y

    def get_edge_type(self, crop_img):
        """Determine direction of ramp edge

        Args:
            crop_img (np.ndarray): cropped pixel array ~ subset of the image

        Returns:
            tuple of string: vertical/horizontal and up/down or left/rigtward
        """
        edge_sum_rows = np.sum(crop_img, axis=1).astype(np.int_)
        edge_sum_cols = np.sum(crop_img, axis=0).astype(np.int_)

        _, pk_rows_height = self.ACR_obj.find_n_highest_peaks(
            np.abs(np.diff(edge_sum_rows)), 1
        )
        _, pk_cols_height = self.ACR_obj.find_n_highest_peaks(
            np.abs(np.diff(edge_sum_cols)), 1
        )

        edge_type = "vertical" if pk_rows_height > pk_cols_height else "horizontal"

        thresh_roi_crop = crop_img > 0.6 * np.max(crop_img)
        edge_dir = (
            np.sum(thresh_roi_crop, axis=0)
            if edge_type == "vertical"
            else np.sum(thresh_roi_crop, axis=1)
        )
        if edge_type == "vertical":
            direction = "downward" if edge_dir[-1] > edge_dir[0] else "upward"
        else:
            direction = "leftward" if edge_dir[-1] > edge_dir[0] else "rightward"

        return edge_type, direction

    def edge_location_for_plot(self, crop_img, edge_type):
        """Determine the location of the edge so it can be visualised

        Args:
            crop_img (np.array): cropped pixel array ~ subset of the image
            edge_type (tuple): vertical/horizontal and up/down or left/rigtward

        Returns:
            np.array: mask array for edge location
        """
        thresh_roi_crop = crop_img > 0.6 * np.max(crop_img)

        naive_lsf = (
            np.abs(np.diff(np.sum(thresh_roi_crop, 1))) > 1
            if edge_type == "vertical"
            else np.abs(np.diff(np.sum(thresh_roi_crop, 0)))
        )
        edge_test = np.diff(np.where(naive_lsf == 0))[0]
        edge_begin = np.where(edge_test > 1)
        edge_loc = np.array(
            [edge_begin, edge_begin + edge_test[edge_begin] - 1]
        ).flatten()

        return edge_loc

    def fit_normcdf_surface(self, crop_img, edge_type, direction):
        """Fit normalised CDF? to surface

        Args:
            crop_img (np.array): cropped pixel array ~ subset of the image
            edge_type (string): vertical/horizontal
            direction (string): up/down or left/rigtward

        Returns:
            tuple of floats: slope, surface
        """
        thresh_roi_crop = crop_img > 0.6 * np.max(crop_img)
        temp_x = np.linspace(1, thresh_roi_crop.shape[1], thresh_roi_crop.shape[1])
        temp_y = np.linspace(1, thresh_roi_crop.shape[0], thresh_roi_crop.shape[0])
        x, y = np.meshgrid(temp_x, temp_y)

        bright = max(crop_img[thresh_roi_crop])
        dark = 20 + np.min(crop_img[~thresh_roi_crop])

        def func(x, slope, mu, bright, dark):
            """Maths function

            Args:
                x (_type_): _description_
                slope (_type_): _description_
                mu (_type_): _description_
                bright (_type_): _description_
                dark (_type_): _description_

            Returns:
                _type_: _description_
            """
            norm_cdf = (bright - dark) * scipy.stats.norm.cdf(
                x[0], mu + slope * x[1], 0.5
            ) + dark

            return norm_cdf

        sign = 1 if direction in ("downward", "leftward") else -1
        x_data = (
            np.vstack((sign * x.ravel(), y.ravel()))
            if edge_type == "vertical"
            else np.vstack((sign * y.ravel(), x.ravel()))
        )

        popt, pcov = scipy.optimize.curve_fit(
            func, x_data, crop_img.ravel(), p0=[0, 0, bright, dark], maxfev=1000
        )
        surface = func(x_data, popt[0], popt[1], popt[2], popt[3]).reshape(
            crop_img.shape
        )

        slope = 1 / popt[0] if direction in ("leftward", "upward") else -1 / popt[0]

        return slope, surface

    def sample_erf(self, crop_img, slope, edge_type):
        """_summary_

        Args:
            crop_img (np.array): cropped pixel array ~ subset of the image
            slope (float): value of slope of edge
            edge_type (string): vertical/horizontal

        Returns:
            np.array: _description_
        """
        resamp_factor = 8
        if edge_type == "horizontal":
            resample_crop_img = cv2.resize(
                crop_img, (crop_img.shape[0] * resamp_factor, crop_img.shape[1])
            )
        else:
            resample_crop_img = cv2.resize(
                crop_img, (crop_img.shape[0], crop_img.shape[1] * resamp_factor)
            )

        mid_loc = [i / 2 for i in resample_crop_img.shape]

        temp_x = np.linspace(1, resample_crop_img.shape[1], resample_crop_img.shape[1])
        temp_y = np.linspace(1, resample_crop_img.shape[0], resample_crop_img.shape[0])
        x_resample, y_resample = np.meshgrid(temp_x, temp_y)

        erf = []
        n_inside_roi = []
        if edge_type == "horizontal":
            diffY = (y_resample - 1) - mid_loc[0]
            x_prime = x_resample + resamp_factor * diffY * slope

            x_min, x_max = np.min(x_prime).astype(int), np.max(x_prime).astype(int)

            for k in range(x_min, x_max):
                erf_val = np.mean(resample_crop_img[(x_prime >= k) & (x_prime < k + 1)])
                erf.append(erf_val)
                number_nonzero = np.count_nonzero(
                    resample_crop_img[(x_prime >= k) & (x_prime < k + 1)]
                )
                n_inside_roi.append(number_nonzero)
        else:
            diffX = (x_resample.shape[0] - 1) - x_resample - mid_loc[1]
            y_prime = np.flipud(y_resample) + resamp_factor * diffX * slope

            y_min, y_max = np.min(y_prime).astype(int), np.max(y_prime).astype(int)

            for k in range(y_min, y_max):
                erf_val = np.mean(resample_crop_img[(y_prime >= k) & (y_prime < k + 1)])
                erf.append(erf_val)
                number_nonzero = np.count_nonzero(
                    resample_crop_img[(y_prime >= k) & (y_prime < k + 1)]
                )
                n_inside_roi.append(number_nonzero)

        erf = np.array(erf)
        n_inside_roi = np.array(n_inside_roi)

        erf = erf[n_inside_roi == np.max(n_inside_roi)]

        return erf

    def fit_erf(self, erf):
        """Fit ERF

        Args:
            erf (np.array): _description_

        Returns:
            _type_: _description_
        """
        true_erf = np.diff(erf) > 0.2 * np.max(np.diff(erf))
        turning_points = np.where(true_erf)[0][0], np.where(true_erf)[0][-1]
        weights = 0.5 * np.ones((len(true_erf) + 1))
        weights[turning_points[0] : turning_points[1]] = 1

        def func(x, a, b, c, d, e):
            """Maths function for sigmoid curve equation

            Args:
                x (_type_): _description_
                a (_type_): _description_
                b (_type_): _description_
                c (_type_): _description_
                d (_type_): _description_
                e (_type_): _description_

            Returns:
                _type_: _description_
            """
            sigmoid = a + b / (1 + np.exp(c * (x - d))) ** e

            return sigmoid

        popt, pcov = scipy.optimize.curve_fit(
            func,
            np.arange(1, len(erf) + 1),
            erf,
            sigma=(1 / weights),
            p0=[np.min(erf), np.max(erf), 0, sum(turning_points) / 2, 1],
            maxfev=5000,
        )
        erf_fit = func(
            np.arange(1, len(erf) + 1), popt[0], popt[1], popt[2], popt[3], popt[4]
        )

        return erf_fit

    def identify_MTF50(self, freq, MTF):
        """Calculate effective resolution

        Args:
            freq (float or int): _description_
            MTF (float or int): _description_

        Returns:
            float: _description_
        """
        freq_interp = np.arange(0, 1.005, 0.005)
        MTF_interp = np.interp(
            freq_interp, freq, MTF, left=None, right=None, period=None
        )
        equivalent_linepairs = freq_interp[np.argmin(np.abs(MTF_interp - 0.5))]
        eff_res = 1 / (equivalent_linepairs * 2)

        return eff_res

    def get_roi_resolution(self, img, width, height, loc, dx, dy):
        roi_x, roi_y = loc

        roi = create_rectangular_roi_at(img, width, height, roi_x, roi_y)
        roi[roi.mask] = 0
        #debug_image_sample(roi)

        non_zero = roi[roi > 0]
        center, window_width = self.ACR_obj.compute_center_and_width(non_zero)
        leveled = self.ACR_obj.apply_window_width_center(roi, center, window_width)
        #debug_image_sample(leveled)

        crop_img = self.ACR_obj.crop_image(leveled, roi_x, roi_y, width)
        #debug_image_sample(crop_img)

        crop_height = crop_img.shape[1]
        #non_zero = crop_img[]
        for col in range(crop_img.shape[0]):
            line_profile = skimage.measure.profile_line(
                crop_img,
                (col, 0),
                (col, crop_height),
                mode="constant",
            ).flatten()

    def get_mtf50(self, dcm):
        """_summary_

        Args:
            dcm (pydicom.Dataset): DICOM image object

        Returns:
            tuple: _description_
        """
        img = dcm.pixel_array
        dx, dy = self.ACR_obj.dx, self.ACR_obj.dy
        cxy, _ = self.ACR_obj.find_phantom_center(img, dx, dy)

        logger.info(f"Center {cxy}")
        logger.info(f"Pixel Resolution {dx, dy}")
        ramp_x, ramp_y = (int(cxy[0] - 17 / dx), int(cxy[1] + 35 / dy))
        logger.info(f"Adjusted Center {ramp_x, ramp_y}")

        width = int(13 * img.shape[0] / 256)
        self.get_roi_resolution(img, width, width, (ramp_x, ramp_y), dx, dy)

        #crop_img = self.ACR_obj.crop_image(img, ramp_x, ramp_y, width)
        #edge_type, direction = self.get_edge_type(crop_img)
        #slope, surface = self.fit_normcdf_surface(crop_img, edge_type, direction)
        #erf = self.sample_erf(crop_img, slope, edge_type)
        #erf_fit = self.fit_erf(erf)

        #freq, lsf_raw, MTF_raw = self.ACR_obj.calculate_MTF(erf, dx, dy)
        #_, lsf_fit, MTF_fit = self.ACR_obj.calculate_MTF(erf_fit, dx, dy)

        #eff_raw_res = self.identify_MTF50(freq, MTF_raw)
        #eff_fit_res = self.identify_MTF50(freq, MTF_fit)

        if self.report:
            self.write_report(dcm, img, ramp_x, ramp_y, width)

        #return eff_raw_res, eff_fit_res
        return 1, 1
