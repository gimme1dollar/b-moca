import os
import time
import subprocess
import xml.etree.ElementTree as ET
from appium.webdriver.common.appiumby import AppiumBy

_WORK_PATH = os.environ["BMOCA_HOME"]


def check_disable_prime(driver):
    command = 'adb shell'
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    process.stdin.write('su\n')
    process.stdin.flush()
    process.stdin.write('cp /data/data/org.wikipedia/shared_prefs/org.wikipedia_preferences.xml /sdcard/org.wikipedia_preferences.xml\n')
    process.stdin.flush()
    time.sleep(0.5)
    command = f'adb pull /sdcard/org.wikipedia_preferences.xml {_WORK_PATH}/bmoca/environment/evaluator_script/wikipedia\n'
    _ = subprocess.run(command, text=True, shell=True)
    
    tree = ET.parse(f'{_WORK_PATH}/bmoca/environment/evaluator_script/wikipedia/org.wikipedia_preferences.xml')
    root = tree.getroot()
    curr_state = root[-1].text
    target_state = '[true,false,false,true,false,true,false,true,true,true]'
    
    try:
        feed_UI = driver.find_element(AppiumBy.ID, 
                                        'org.wikipedia:id/nav_tab_explore')
        in_feed = feed_UI.get_attribute("selected")
    except Exception:
        return False
    
    return curr_state == target_state and in_feed == 'true'