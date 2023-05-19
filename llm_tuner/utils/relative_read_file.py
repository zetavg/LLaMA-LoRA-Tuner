import os


def relative_read_file(base_file, relative_path):
    src_dir = os.path.dirname(os.path.abspath(base_file))
    file_path = os.path.join(src_dir, relative_path)
    with open(file_path, 'r') as f:
        file_contents = f.read()
        return file_contents
