import numpy as np

ascii_chars = " A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)=".lower()
braille_chars = "⠀⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯⠰⠱⠲⠳⠴⠵⠶⠷⠸⠹⠺⠻⠼⠽⠾⠿"


class BrailleConfig:
    DOT_DIAMETER = 1.5 # mm
    DOT_HEIGHT = 0.6 # mm
    DOT_SPACING = 2.3 # mm
    CELL_SPACING_X = 6.1 # mm
    CELL_SPACING_Y = 10 # mm

    CELL_GAP_X = CELL_SPACING_X - DOT_SPACING
    CELL_GAP_Y = CELL_SPACING_Y - DOT_SPACING * 2

    INCLUDE_CAPS = True # Should the braille display capital letters, or should they all be lower


def str_to_braille(string:str) -> str:
    braille_string = ""

    for char in string:
        if char.lower() in ascii_chars:
            # Handle capital letters
            if char.isupper():
                braille_string += braille_chars[ascii_chars.index(",")] + " "
            
            # Handle letter
            braille_string += braille_chars[ascii_chars.index(char.lower())] + " "

        else:
            raise Exception(f"Ascii character \"{char}\" can't be converted to braille.")
        
    return braille_string

def char_to_dots(char:str, position:np.ndarray, config:BrailleConfig) -> np.ndarray:
    points = []

    char = char.lower()

    if char in ascii_chars:
        binary_string = f"{ascii_chars.index(char):06b}" # eg 5 -> 000101
        binary_string = binary_string[::-1] # Get lsb first

        for i in range(6):
            if binary_string[i] == "1":
                points.append(np.array([
                    np.floor_divide(i, 3),
                    -(i % 3),
                    0]) * config.DOT_SPACING + position)

    else:
        raise Exception(f"Ascii character \"{char}\" can't be converted to braille.")

    return points

def str_to_dots(string:str,
                start_position:np.ndarray=np.array([0, 0, 0]),
                rotation:float=0,
                config:BrailleConfig=BrailleConfig()) -> np.ndarray:
    points = []

    position = start_position.copy()

    for char in string:
        if char == "\n":
            position[0] = start_position[0]
            position[1] += config.CELL_SPACING_Y

        else:
            # Handle capital letters
            if char.isupper() and config.INCLUDE_CAPS:
                points += char_to_dots(",", position, config)
                position[0] += config.CELL_SPACING_X

            # Add letter
            points += char_to_dots(char, position, config)
            position[0] += config.CELL_SPACING_X
    
    return np.array(points)


if __name__ == "__main__":
    from geometry_utils import plot_3d_path

    string = "Cambridge Surprise Major"

    print(str_to_braille(string))
    plot_3d_path(str_to_dots(string))