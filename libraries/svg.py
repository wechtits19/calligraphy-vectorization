"""
Library for the purpose of simply creating SVG documents with only path elements.
A path can represent Bézier curves, lines, etc. See SVG specification for further details.
The given functions need to be used in order:
 1. create_file
 2. add_path_element - for as many paths as desired
 3. save_file
"""


def create_file(filename, width, height):
    """
    Create a SVG file of specified width, height.
    """
    f = open(filename, 'w')
    f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
    f.write(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink='
            f'"http://www.w3.org/1999/xlink" baseProfile="tiny" height="{height}" version="1.2" width="{width}">\n')
    f.write('<defs/>\n')
    return f


def add_path_element(file, path_string):
    """
    Add a path element to a previously created SVG file.
    """
    file.write(f'<path d="{path_string}"/>')


def save_file(file):
    """
    Save the previously created SVG file.
    """
    file.write('</svg>')
    file.close()
