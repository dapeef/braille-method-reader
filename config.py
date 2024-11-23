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

    dot_type = BrailleDotType.CYLINDER

class LineConfig:
    width = 1.5 # mm
    height = 1.5 # mm
    dot_separation = 2.3 # mm
    cross_section = PathCrossSection.CYLINDER

class PlateConfig:
    margin = 5 # mm - Border around the edges
    base_type = BaseType.HOLE
    plate_thickness = .4 # mm

    unit_width = 5 # mm
    unit_height = 2.5 # mm

    thick_line_config = LineConfig()
    thick_line_config.width = 1.5 # mm
    thick_line_config.height = 1 # mm
    thick_line_config.cross_section = PathCrossSection.CYLINDER

    thin_line_config = LineConfig()
    thin_line_config.width = 1 # mm
    thin_line_config.height = .5 # mm
    thin_line_config.cross_section = PathCrossSection.CYLINDER

    treble_line_config = LineConfig()
    treble_line_config.width = 1.5 # mm
    treble_line_config.height = 0.6 # mm
    treble_line_config.dot_separation = 2.3 # mm
    treble_line_config.cross_section = PathCrossSection.CYLINDER # Only used if treble_type is "cross"
    treble_type = TrebleType.NONE

    half_lead_line_config = LineConfig()
    half_lead_line_config.width = 1.6 # mm
    half_lead_line_config.height = 0.8 # mm
    half_lead_line_config.dot_separation = 2.5 # mm
    half_lead_line_config.cross_section = PathCrossSection.CYLINDER

    lead_end_marker_type = LeadEndMarkerType.NONE
    lead_end_marker_width = 2 * thick_line_config.width # mm
    lead_end_marker_height = 2 * thick_line_config.height # mm
    lead_end_t_overflow = .5 * margin

    length_type = LengthTypes.SINGLE_LEAD

    reverse_method = False

    title_position = TitlePos.CENTER_VERTICAL
    title_language = TitleLanguage.BOTH
    title_text = TitleText.SHORT

    place_enlargement = 2 # How much longer to draw the places than the dodges
    # normalise_length = True # Whether to squash the line to fit in the same space as if place_enlargement = 1

    braille_config = BrailleConfig()
