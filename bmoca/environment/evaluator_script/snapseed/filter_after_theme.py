import os
import json
import time
import subprocess
import xml.etree.ElementTree as ET
from appium.webdriver.common.appiumby import AppiumBy

_WORK_PATH = os.environ["BMOCA_HOME"]


def check_apply_filter_after_theme(driver):
    try:
        command = "adb shell"
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        )
        process.stdin.write("su\n")
        process.stdin.flush()
        process.stdin.write(
            "cp /data/data/com.niksoftware.snapseed/shared_prefs/Preferences.xml /sdcard/Preferences.xml\n"
        )
        process.stdin.flush()
        time.sleep(0.5)
        command = f"adb pull /sdcard/Preferences.xml {_WORK_PATH}/bmoca/environment/evaluator_script/snapseed\n"
        _ = subprocess.run(command, text=True, shell=True)

        tree = ET.parse(
            f"{_WORK_PATH}/bmoca/environment/evaluator_script/snapseed/Preferences.xml"
        )
        root = tree.getroot()

        for elem in root.findall("boolean"):
            if elem.get("name") == "pref_appearance_use_dark_theme":
                dark_theme_value = elem.get("value")
                break

        theme_set = dark_theme_value == "true"

        s03_UI = driver.find_element(
            AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().text("S03")'
        )

        return theme_set and s03_UI.get_attribute("selected") == "true"

    except:
        return False
