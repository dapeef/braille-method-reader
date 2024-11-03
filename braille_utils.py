import numpy as np
from config import BrailleConfig

ascii_chars = " A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)=".lower()
braille_chars = " ⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯⠰⠱⠲⠳⠴⠵⠶⠷⠸⠹⠺⠻⠼⠽⠾⠿"
numbers = "0123456789"


def str_to_braille(string:str, legible:bool=False) -> str:
    braille_string = ""
    was_previous_number = False

    for char in string:
        if char.lower() in ascii_chars:
            # Handle prefix characters
            if char.isupper():
                braille_string += braille_chars[ascii_chars.index(",")]

            if char in numbers:
                if not was_previous_number:
                    braille_string += braille_chars[ascii_chars.index("#")]
                was_previous_number = True
            else:
                was_previous_number = False

            # Handle letter
            braille_string += braille_chars[ascii_chars.index(char.lower())]

        else:
            raise Exception(f"Ascii character \"{char}\" can't be converted to braille.")
    
    if legible:
        legible_string = ""
        for char in braille_string:
            legible_string += char + " "
        
        braille_string = legible_string[:-1]

        
    return braille_string

def char_to_dots(braille_char:str, position:np.ndarray, config:BrailleConfig) -> np.ndarray:
    points = []

    binary_string = f"{braille_chars.index(braille_char):06b}" # eg 5 -> 000101
    binary_string = binary_string[::-1] # Get lsb first

    for i in range(6):
        if binary_string[i] == "1":
            points.append(np.array([
                np.floor_divide(i, 3),
                -(i % 3),
                0]) * config.dot_spacing + position)

    return points

def wrap_text(text, max_chars_per_line):
    """
    Wrap text to a specified maximum line length.
    
    Preferentially splits on spaces. If a single word is longer than 
    max_chars_per_line, it will be split at max_chars_per_line.
    
    Args:
        text (str): The input text to be wrapped
        max_chars_per_line (int): Maximum number of characters per line
    
    Returns:
        str: The wrapped text with lines no longer than max_chars_per_line
    """
    # Handle empty string case
    if not text:
        return ""
    
    # Split the text into words
    words = text.split(" ")
    
    # List to store the wrapped lines
    wrapped_lines = []
    
    # Current line being constructed
    current_line = []
    current_line_length = 0
    
    for word in words:
        # Check if adding this word would exceed max line length
        # Include a space before the word (except for first word)
        word_length = len(word)
        space_needed = 1 if current_line else 0
        
        # If word is longer than max line length, force split
        if word_length > max_chars_per_line:
            # If current line is not empty, add it first
            if current_line:
                wrapped_lines.append(' '.join(current_line))
                current_line = []
                current_line_length = 0
            
            # Split long word across multiple lines
            for i in range(0, word_length, max_chars_per_line):
                wrapped_lines.append(word[i:i+max_chars_per_line])
        
        # Check if adding this word would exceed max line length
        elif (current_line_length + space_needed + word_length) > max_chars_per_line:
            # Current line is full, add to wrapped lines
            wrapped_lines.append(' '.join(current_line))
            current_line = [word]
            current_line_length = word_length
        
        else:
            # Add word to current line
            # Add a space if not the first word on the line
            if current_line:
                current_line_length += 1  # for the space
            current_line.append(word)
            current_line_length += word_length
    
    # Add any remaining line
    if current_line:
        wrapped_lines.append(' '.join(current_line))
    
    return '\n'.join(wrapped_lines)

def str_to_dots(string:str,
                config:BrailleConfig=BrailleConfig(),
                max_line_length=np.inf) -> np.ndarray:
    points = []
    position = np.array([0, 0, 0])

    if not config.include_caps:
        string = string.lower()

    braille_string = str_to_braille(string)

    max_chars_per_line = np.floor((max_line_length + config.cell_gap_x) / config.cell_spacing_x)
    braille_string = wrap_text(braille_string, max_chars_per_line)

    for char in braille_string:
        if char == "\n":
            position[0] = 0
            position[1] -= config.cell_spacing_y

        else:
            # Add letter
            points += char_to_dots(char, position, config)
            position[0] += config.cell_spacing_x
    
    return np.array(points)


if __name__ == "__main__":
    from geometry_utils import plot_3d_path

    string = "Cambridge 888"

    print(str_to_braille(string, legible=True))
    plot_3d_path(str_to_dots(string))
