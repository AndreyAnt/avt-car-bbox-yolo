import unittest

from car_detection_logic import _vehicle_class_ids


class VehicleClassTests(unittest.TestCase):
    def test_coco_vehicles_include_cars_buses_and_trucks(self):
        names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck", 8: "boat"}
        self.assertEqual(_vehicle_class_ids(names), [2, 5, 7])

    def test_class_ids_follow_metadata_including_explicit_van_category(self):
        names = {7: "person", 9: "van", 0: "truck", 6: "car", 4: "bus"}
        self.assertEqual(_vehicle_class_ids(names), [0, 4, 6, 9])

    def test_models_with_only_a_supported_vehicle_category_are_accepted(self):
        for label in ("car", "truck", "bus", "van"):
            with self.subTest(label=label):
                self.assertEqual(_vehicle_class_ids({0: label, 2: "person"}), [0])

    def test_unknown_model_does_not_fall_back_to_arbitrary_coco_ids(self):
        with self.assertRaisesRegex(ValueError, "no supported vehicle classes"):
            _vehicle_class_ids({2: "person", 5: "dog", 7: "cat"})


if __name__ == "__main__":
    unittest.main()
