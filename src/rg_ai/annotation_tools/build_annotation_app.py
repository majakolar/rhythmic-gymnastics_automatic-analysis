import os
import subprocess
import sys

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_SCRIPT = os.path.join('src', 'rg_ai', 'annotation_tools', 'annotate_db.py')
DB_CONFIG = os.path.join('src', 'rg_ai', 'annotation_tools', 'db_config.json')
ICON = os.path.join('src', 'rg_ai', 'annotation_tools', 'icon.ico')

def run(cmd, check=True, env=None):
    print(f"\n[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, env=env)
    return result

# dependencies in Wine Python
print("Installing dependencies in Wine Python...")
run('wine "C:/Program Files/Python39/python.exe" -m pip install -r requirements.txt')

# windows executable with PyInstaller
print("Building Windows executable with PyInstaller...")
add_data = f'{DB_CONFIG};rg_ai/annotation_tools'
icon_opt = f'--icon={ICON}' if os.path.exists(ICON) else ''
print(icon_opt)

pyi_cmd = (
    f'wine "C:/Program Files/Python39/Scripts/pyinstaller.exe" '
    f'--onefile --windowed --clean --noconfirm '
    f'--name=rg_annotation '
    f'--add-data "{add_data}" '
    f'{icon_opt} '
    f'{MAIN_SCRIPT}'
)
run(pyi_cmd)

print(f"Build complete! Check the 'dist' directory for rg_annotation.exe.") 