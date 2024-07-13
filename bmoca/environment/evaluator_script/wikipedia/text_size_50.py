import os
import time
import subprocess
import xml.etree.ElementTree as ET
from appium.webdriver.common.appiumby import AppiumBy

_WORK_PATH = os.environ["BMOCA_HOME"]


def check_text_size_50(driver):
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
            "cp /data/data/org.wikipedia/shared_prefs/org.wikipedia_preferences.xml /sdcard/org.wikipedia_preferences.xml\n"
        )
        process.stdin.flush()
        time.sleep(0.5)
        command = f"adb pull /sdcard/org.wikipedia_preferences.xml {_WORK_PATH}/bmoca/environment/evaluator_script/wikipedia\n"
        _ = subprocess.run(command, text=True, shell=True)

        tree = ET.parse(
            f"{_WORK_PATH}/bmoca/environment/evaluator_script/wikipedia/org.wikipedia_preferences.xml"
        )
        root = tree.getroot()
        text_size_multiplier = root.find(".//int[@name='textSizeMultiplier']").attrib[
            "value"
        ]

        return text_size_multiplier == "-5"
    except:
        return False
