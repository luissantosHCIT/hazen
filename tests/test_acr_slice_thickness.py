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
    centers = [(75.0, 2.0), (77.0, 2.0)]
    dz = 5.44
    dz_3x = 7.97
    top_dz = 5.2
    bottom_dz = 5.7

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_slice_thickness_task = ACRSliceThickness(input_data=input_files)

        self.dcm = self.acr_slice_thickness_task.ACR_obj.slice_stack[0]
        self.centre, _ = self.acr_slice_thickness_task.ACR_obj.find_phantom_center(
            self.dcm.pixel_array, self.dcm.PixelSpacing[0], self.dcm.PixelSpacing[1]
        )
        self.results = self.acr_slice_thickness_task.get_slice_thickness(self.dcm)
        self.large_thickness = self.acr_slice_thickness_task.compute_thickness(
            self.results['ramps']['top']['width'],
            self.results['ramps']['bottom']['width'] * 3
        )
        self.top_thickness = self.acr_slice_thickness_task.compute_thickness(
            self.results['ramps']['top']['width'],
            self.results['ramps']['top']['width']
        )
        self.bottom_thickness = self.acr_slice_thickness_task.compute_thickness(
            self.results['ramps']['bottom']['width'],
            self.results['ramps']['bottom']['width']
        )

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

    def test_exagerated_slice_thickness(self):
        """ This is test is meant to ensure that the formula was implemented properly. """
        slice_thickness_val = round(
            self.large_thickness, 2
        )

        print("\ntest_slice_thickness.py::TestSliceThickness::test_exagerated_slice_thickness")
        print("new_release_value:", slice_thickness_val)
        print("fixed_value:", self.dz_3x)

        assert slice_thickness_val == self.dz_3x

    def test_top_ramp_only(self):
        """ This is test is meant to ensure that the formula was implemented properly. """
        slice_thickness_val = round(
            self.results['thickness'], 2
        )
        top_slice_thickness_val = round(
            self.top_thickness, 2
        )

        print("\ntest_slice_thickness.py::TestSliceThickness::test_top_ramp_only")
        print("new_release_value:", slice_thickness_val)
        print("new_release_value (top only):", top_slice_thickness_val)
        print("fixed_value:", self.top_dz)

        assert not (slice_thickness_val == top_slice_thickness_val and slice_thickness_val == self.top_dz)

    def test_bottom_ramp_only(self):
        """ This is test is meant to ensure that the formula was implemented properly. """
        slice_thickness_val = round(
            self.results['thickness'], 2
        )
        bottom_slice_thickness_val = round(
            self.bottom_thickness, 2
        )

        print("\ntest_slice_thickness.py::TestSliceThickness::test_top_ramp_only")
        print("new_release_value:", slice_thickness_val)
        print("new_release_value (bottom only):", bottom_slice_thickness_val)
        print("fixed_value:", self.bottom_dz)

        assert not (slice_thickness_val == bottom_slice_thickness_val and slice_thickness_val == self.bottom_dz)


class TestACRSliceThicknessPhilipsAchieva(TestACRSliceThicknessSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    centers = [(79.0, 2.0), (72.0, 2.0)]
    dz = 5.59
    dz_3x = 7.75
    top_dz = 4.8
    bottom_dz = 6.7


class TestACRSliceThicknessSiemensSolaFit(TestACRSliceThicknessSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    centers = [(75.0, 2.0), (80.0, 2.0)]
    dz = 5.03
    dz_3x = 7.29
    top_dz = 4.7
    bottom_dz = 5.4
