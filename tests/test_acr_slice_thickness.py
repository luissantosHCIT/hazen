import os
import unittest
import pathlib
import pydicom

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR


class TestACRSliceThicknessSiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    centers = [(44.0, 2.0), (46.0, 2.0)]
    dz = 5.18

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_slice_thickness_task = ACRSliceThickness(input_data=input_files)

        self.dcm = self.acr_slice_thickness_task.ACR_obj.slice_stack[0]
        self.centre, _ = self.acr_slice_thickness_task.ACR_obj.find_phantom_center(
            self.dcm.pixel_array, self.dcm.PixelSpacing[0], self.dcm.PixelSpacing[1]
        )
        self.results = self.acr_slice_thickness_task.get_slice_thickness(self.dcm)

    def test_slice_thickness(self):
        slice_thickness_val = round(
            self.results['thickness'], 2
        )

        print("\ntest_slice_thickness.py::TestSliceThickness::test_slice_thickness")
        print("new_release_value:", slice_thickness_val)
        print("fixed_value:", self.dz)

        assert slice_thickness_val == self.dz

    def test_ramp_slot_relative_centers(self):
        centers = [self.results['ramps']['top']['center'], self.results['ramps']['bottom']['center']]

        print("\ntest_slice_thickness.py::TestSliceThickness::test_ramp_slot_relative_centers")
        print("new_release_value:", centers)
        print("fixed_value:", self.centers)

        assert centers == self.centers


class TestACRSliceThicknessPhilipsAchieva(TestACRSliceThicknessSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    centers = [(46.0, 2.0), (39.0, 2.0)]
    dz = 4.69


class TestACRSliceThicknessSiemensSolaFit(TestACRSliceThicknessSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    centers = [(44.0, 2.0), (48.0, 2.0)]
    dz = 5.18
