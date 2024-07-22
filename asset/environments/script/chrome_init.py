import time
import subprocess
from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy


def chrome_init(avd_name="pixel_3_train_00"):
    options = AppiumOptions()
    options.load_capabilities(
        {
            "platformName": "Android",
            "appium:platformVersion": "10",
            "appium:deviceName": avd_name,
            "appium:automationName": "UiAutomator2",
            "appium:appPackage": "com.android.chrome",
            "appium:appActivity": "com.google.android.apps.chrome.Main",
            "appium:ensureWebviewsHavePages": True,
            "appium:nativeWebScreenshot": True,
            "appium:newCommandTimeout": 3600,
            "appium:connectHardwareKeyboard": True,
        }
    )
    driver = webdriver.Remote("http://127.0.0.1:4723", options=options)
    time.sleep(2)

    attempt = 0
    while attempt < 5:
        try:
            terms_accept_UI = driver.find_element(
                by=AppiumBy.ID, value="com.android.chrome:id/terms_accept"
            )
            terms_accept_UI.click()
            time.sleep(1)
            break
        except:
            time.sleep(1)
            attempt += 1

    attempt = 0
    while attempt < 5:
        try:
            skip_signin_UI = driver.find_element(
                by=AppiumBy.ID, value="com.android.chrome:id/negative_button"
            )
            skip_signin_UI.click()
            time.sleep(1)
            break
        except:
            time.sleep(1)
            attempt += 1

    command = f"adb shell input keyevent 3"  # home button
    result = subprocess.run(command, text=True, shell=True)
    time.sleep(1)

    print("Chrome app initialized successfully.")
