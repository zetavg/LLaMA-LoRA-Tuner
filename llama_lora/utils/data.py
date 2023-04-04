import os
import shutil
import fnmatch

from ..globals import Global


def init_data_dir():
    current_file_path = os.path.abspath(__file__)
    parent_directory_path = os.path.dirname(current_file_path)
    project_dir_path = os.path.abspath(
        os.path.join(parent_directory_path, "..", ".."))
    copy_sample_data_if_not_exists(os.path.join(project_dir_path, "templates"),
                                   os.path.join(Global.data_dir, "templates"))


def copy_sample_data_if_not_exists(source, destination):
    if os.path.exists(destination):
        return

    print(f"Copying sample data to \"{destination}\"")
    shutil.copytree(source, destination)


def get_available_template_names():
    templates_directory_path = os.path.join(Global.data_dir, "templates")
    all_files = os.listdir(templates_directory_path)
    return [os.path.splitext(filename)[0] for filename in all_files if fnmatch.fnmatch(filename, "*.json")]
