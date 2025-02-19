"""
ACR Geometric Accuracy

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates geometric accuracy for slices 1 and 5 of the ACR phantom.

This script calculates the horizontal and vertical lengths of the ACR phantom in Slice 1 in accordance with the ACR
Guidance.
This script calculates the horizontal, vertical and diagonal lengths of the ACR phantom in Slice 5 in accordance with
the ACR Guidance.
The average distance measurement error, maximum distance measurement error and coefficient of variation of all distance
measurements is reported as recommended by IPEM Report 112, "Quality Control and Artefacts in Magnetic Resonance
Imaging".

This is done by first producing a binary mask for each respective slice. Line profiles are drawn with aid of rotation
matrices around the centre of the test object to determine each respective length. The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

18/11/2022
"""

import os
import sys
import traceback

import cv2
import numpy as np

import skimage.measure
import skimage.transform
import skimage.morphology

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib import logger
from hazenlib.utils import debug_image_sample


class ACRSagittalGeometricAccuracy(HazenTask):
    """Geometric accuracy measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing geometric accuracy measurement using the first and fifth slices from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation.
        """
        dcm = self.ACR_obj.slice_stack[0]
        img_desc = self.img_desc(dcm)

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = [
            img_desc
        ]

        try:
            lengths_1 = self.get_geometric_accuracy(dcm)
            logger.info(lengths_1)
            results["measurement"][img_desc] = {
                "Vertical distance": round(lengths_1, 2)
            }
        except Exception as e:
            logger.error(
                f"Could not calculate the geometric accuracy for {self.img_desc(self.ACR_obj.slice_stack[0])} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def write_report(self, img, dcm, length_dict, mask, cxy, offset):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1)
        fig.set_size_inches(8, 24)
        fig.tight_layout(pad=4)

        axes[0].imshow(img)
        axes[0].scatter(cxy[0], cxy[1], c="red")
        axes[0].set_title("Centroid Location")

        axes[1].imshow(mask)
        axes[1].set_title("Thresholding Result")

        axes[2].imshow(img)
        axes[2].arrow(
            cxy[0] + offset[0],
            length_dict["Vertical Extent"][0],
            1 + offset[1],
            length_dict["Vertical Extent"][-1]
            - length_dict["Vertical Extent"][0],
            color="orange",
            length_includes_head=True,
            head_width=5,
        )
        axes[2].legend(
            [
                str(np.round(length_dict["Vertical Distance"], 2)) + "mm",
            ]
        )
        axes[2].axis("off")
        axes[2].set_title("Geometric Accuracy for Slice 1")

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
        )
        fig.savefig(img_path)
        self.report_files.append(img_path)


    def get_geometric_accuracy(self, dcm):
        """Measure geometric accuracy for input slice. \n
        Creates a mask over the phantom from the pixel array of the DICOM image.
        Uses the centre and shape of the mask to determine horizontal and vertical lengths,
        and also diagonal lengths in slice 5.

        Args:
            slice_index (int): the index of the slice position, for example slice 5 is at index 4.

        Returns:
            tuple of float: horizontal and vertical distances.
        """
        img, rescaled, presentation = self.ACR_obj.get_presentation_pixels(dcm)

        cxy, _ = self.ACR_obj.find_phantom_center(img, self.ACR_obj.dx, self.ACR_obj.dy, False)
        mask = self.ACR_obj.get_mask_image(img, cxy)

        offset = (int(np.round(-15 / self.ACR_obj.dx)), 0)

        length_dict = self.ACR_obj.measure_orthogonal_lengths(mask, cxy, v_offset=offset)

        if self.report:
            self.write_report(img, dcm, length_dict, mask, cxy, offset)

        return length_dict["Vertical Distance"]
