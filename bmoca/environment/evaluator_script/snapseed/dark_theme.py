import os
import json
import time
import subprocess
import xml.etree.ElementTree as ET

_WORK_PATH = os.environ['BMOCA_HOME']

def check_dark_theme(driver):
    
    command = 'adb shell'
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    process.stdin.write('su\n')
    process.stdin.flush()
    process.stdin.write('cp /data/data/com.niksoftware.snapseed/shared_prefs/Preferences.xml /sdcard/Preferences.xml\n')
    process.stdin.flush()
    time.sleep(0.5)
    command = f'adb pull /sdcard/Preferences.xml {_WORK_PATH}/bmoca/environment/evaluator_script/snapseed\n'
    _ = subprocess.run(command, text=True, shell=True)
    
    tree = ET.parse(f'{_WORK_PATH}/bmoca/environment/evaluator_script/snapseed/Preferences.xml')
    root = tree.getroot()
    
    try: 
        for elem in root.findall('boolean'):
            if elem.get('name') == 'pref_appearance_use_dark_theme':
                dark_theme_value = elem.get('value')
                break
    except Exception:
        return False
    
    return dark_theme_value == 'true'