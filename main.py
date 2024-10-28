from geometry_utils import Plate, PlateConfig, TitlePos, TitleLanguage, TitleText
from method_utils import Method


config = PlateConfig()
config.unit_thickness = .4 # mm
config.unit_width = 10 # mm
config.unit_height = 5 # mm

config.title_position = TitlePos.LEFT 
config.title_language = TitleLanguage.BOTH
config.title_text = TitleText.SHORT

# Norwich 6 = 14317
# Cambridge 8 = 16694
# Cambridge 10 = 21250
# Double Norwich 8 = 12470
method = Method.create_from_complib_id("16694")
# method.name = "D N'ich"

plate = Plate(method, config)

plate.create_base()
plate.create_title()
plate.create_vertical_lines()
plate.create_thick_line(4)
plate.create_place_bell_label(4)
plate.create_dotted_line()

plate.save_to_stl()
