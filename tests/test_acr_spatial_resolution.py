import os
import unittest
import pathlib
import pydicom
import numpy as np

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_spatial_resolution import ACRSpatialResolution
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRSpatialResolutionSiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_spatial_resolution_task = ACRSpatialResolution(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

        self.dcm = self.acr_spatial_resolution_task.ACR_obj.slice_stack[0]

    def test_get_best_resolution(self):
        pass

    def test_get_resolved_row(self):
        pass

    def get_spatially_resolved_rows(self):
        pass


class TestACRSpatialResolutionGE(TestACRSpatialResolutionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    centre = (254, 255)
    rotation_angle = 0
    y_ramp_pos = 244
    width = 26
    edge_type = "vertical", "upward"
    edge_loc = [5, 7]
    slope = 0.037
    MTF50 = (0.72, 0.71)
