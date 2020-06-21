import unittest
from numpy import array
from gds_to_png import WHITE_COLOR, Simulation, Image


class MockGdsObject(object):
    ''' This has elements, and elements have bounding_box'''
    def __init__(self, num_elements, bounding_box):
        self.elements = []
        self.bounding_box = bounding_box
        for i in range(num_elements):
            self.elements.append(MockGdsObject(0, None))

class TestSimulationComponents(unittest.TestCase):
    def test_adding_pores(self):
        image = Image.new("1", (10, 10), color=WHITE_COLOR)
        gds_object = MockGdsObject(2, ((0,0), (10, 10)))
        gds_object.elements[0].bounding_box = array(((0,0), (1,1)))
        gds_object.elements[1].bounding_box = array(((5, 5), (6, 6)))
        pore_points = Simulation.add_pores(image, gds_object, quality_factor=1.0, int_min_x=0, int_min_y=0)
        self.assertEqual(8, len(pore_points))
