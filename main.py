from plate import Plate
from config import PlateConfig
from option_types import *
from method_utils import Method


config = PlateConfig()

config.plate_thickness = .4 # mm

config.title_text = TitleText.FULL

config.braille_config.dot_type = BrailleDotType.CYLINDER

config.reverse_method = False


# Plain bob 5 = 10550
# Norwich 6 = 14317
# Cambridge 8 = 16694
# Cambridge 10 = 21250
# Double Norwich 8 = 12470
# Mareham 8 = 25093
method = Method.create_from_complib_id("25093")
# method.name = "C'bridge"

# # Method plate
# plate = Plate(method, config)
# plate.create_base()
# plate.create_vertical_lines()
# bell = 2
# plate.create_thick_line(bell)
# plate.create_lead_end_markers(bell)
# plate.create_treble_line(blue_line_bell=bell)
# plate.create_half_lead_lines()
# plate.save_to_stl("./stl-files/mk-3/type-7.stl")

# Title plate
plate = Plate(method, config)
plate.create_base()
plate.create_title()
plate.save_to_stl("stl-files/title.stl")
