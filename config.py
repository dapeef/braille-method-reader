from option_types import *

class BrailleConfig:
    dot_diameter = 1.6 # mm
    dot_height = 0.9 # mm
    dot_spacing = 2.5 # mm
    cell_spacing_x = 7.6 # mm
    cell_spacing_y = 10.2 # mm

    cell_gap_x = cell_spacing_x - dot_spacing
    cell_gap_y = cell_spacing_y - dot_spacing * 2

    include_caps = True # Should the braille display capital letters, or should they all be lower

class LineConfig:
    width = 1.5 # mm
    height = 0.6 # mm
    dot_separation = 2.3 # mm
    cross_section = PathCrossSection.CYLINDER

class PlateConfig:
    plate_thickness = .4 # mm

    unit_width = 5 # mm
    unit_height = 2.5 # mm

    thick_line_config = LineConfig()
    thick_line_config.width = 1.5 # mm
    thick_line_config.height = 1.5 # mm
    thick_line_config.cross_section = PathCrossSection.RECTANGLE

    thin_line_config = LineConfig()
    thin_line_config.width = 1 # mm
    thin_line_config.height = .5 # mm
    thin_line_config.cross_section = PathCrossSection.CYLINDER

    treble_line_config = LineConfig()
    treble_line_config.width = 1.5 # mm
    treble_line_config.height = 0.6 # mm
    treble_line_config.dot_separation = 2.3 # mm
    treble_line_config.cross_section = PathCrossSection.CYLINDER # Only used if treble_type is "cross"
    treble_type = TrebleType.DOTTED

    half_lead_line_config = LineConfig()
    half_lead_line_config.width = 1.6 # mm
    half_lead_line_config.height = 0.8 # mm
    half_lead_line_config.dot_separation = 2.5 # mm
    half_lead_line_config.cross_section = PathCrossSection.CYLINDER

    lead_end_dot_diameter = 2 * thick_line_config.width # mm
    lead_end_dot_height = 2 * thick_line_config.height # mm

    length_type = LengthTypes.SINGLE_LEAD

    reverse_method = False

    title_position = TitlePos.CENTER_VERTICAL
    title_language = TitleLanguage.BOTH
    title_text = TitleText.SHORT

    margin = 5 # mm - Border around the edges
    base_type = BaseType.HOLE

    braille_config = BrailleConfig()
