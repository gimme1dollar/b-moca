import os
import json
import time
import subprocess
import xml.etree.ElementTree as ET
from appium.webdriver.common.appiumby import AppiumBy

_WORK_PATH = os.environ['BMOCA_HOME']

def check_apply_filter_after_quality(driver):
    
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
        for elem in root.findall('string'):
            if elem.get('name') == 'pref_export_setting_compression':
                format_quality = elem.text
                break
    except Exception:
        return False
    
    quality_set = format_quality == '100'
    
    try:
        s03_UI = driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR , 'new UiSelector().text("S03")')
    except:
        return False
    
    return quality_set and s03_UI.get_attribute("selected") == 'true'