import os
import json
import time
import subprocess
import xml.etree.ElementTree as ET
from appium.webdriver.common.appiumby import AppiumBy

_WORK_PATH = os.environ['BMOCA_HOME']

def check_go_to_my_profile(driver):
    # # get xml from avd
    # command = f'adb pull /data/data/com.instagram.android/shared_prefs/com.instagram.android_preferences.xml {_WORK_PATH}/bmoca/environment/evaluator_script/tmp'
    # result = subprocess.run(command, text=True, shell=True)
    # time.sleep(1)

    # # load xml file and get user id
    # tree = ET.parse(f'{_WORK_PATH}/bmoca/environment/evaluator_script/tmp/com.instagram.android_preferences.xml')
    # root = tree.getroot()
    # for i in range(len(root)):
    #     if root[i].attrib['name'] == 'current':
    #         current = root[i]
    #         break
    # dict = json.loads(current.text)
    # userid = dict['username']

    # # go to my profile by search
    # try:
    #     element = driver.find_element(AppiumBy.XPATH, f'//android.widget.TextView[@content-desc="{userid}"][@resource-id="com.instagram.android:id/action_bar_title"]')
    #     search_criteria = True
    # except:
    #     search_criteria = False
    
    try:
        profile_tab_UI = driver.find_element(AppiumBy.ID, 
                                         'com.instagram.android:id/profile_tab')
        in_profile_tab = profile_tab_UI.get_attribute("selected")
        return in_profile_tab == 'true'
    except Exception:
        return False
