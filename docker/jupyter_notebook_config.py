import os
import subprocess

def export_script_and_view(model, os_path, contents_manager):
    if model["type"] != "notebook":
        return
    dir_name, file_name = os.path.split(os_path)
    file_base, file_ext = os.path.splitext(file_name)
    if file_base.startswith("Untitled"):
        return
    export_name = file_base if file_ext == ".ipynb" else file_name
    subprocess.check_call(["jupyter", "nbconvert", "--to", "script", file_name, "--output", export_name + "_script"], cwd=dir_name)
    subprocess.check_call(["jupyter", "nbconvert", "--to", "html", file_name, "--output", export_name + "_view"], cwd=dir_name)

c.FileContentsManager.post_save_hook = export_script_and_view
