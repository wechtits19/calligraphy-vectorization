def create_file(filename, width, height):
    f = open(filename, 'w')
    f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
    f.write(
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" baseProfile="tiny" height="{height}" version="1.2" width="{width}">\n')
    f.write('<defs/>\n')
    return f


def add_path_element(file, path_string):
    file.write(f'<path d="{path_string}"/>')


def save_file(file):
    file.write('</svg>')
    file.close()
