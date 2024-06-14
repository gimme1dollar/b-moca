import time
import subprocess
from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy

def snapseed_init(avd_name='pixel_3_train_00'):
    options = AppiumOptions()
    options.load_capabilities({
        "platformName": "Android",
        "appium:platformVersion": "10",
        "appium:deviceName": avd_name,
        "appium:automationName": "UiAutomator2",
        "appium:appPackage": "com.niksoftware.snapseed",
        "appium:appActivity": "com.google.android.apps.snapseed.MainActivity",
        "appium:ensureWebviewsHavePages": True,
        "appium:nativeWebScreenshot": True,
        "appium:newCommandTimeout": 3600,
        "appium:connectHardwareKeyboard": True
    })

    driver = webdriver.Remote("http://127.0.0.1:4723", options=options)
    time.sleep(2)

    attempt = 0
    while attempt < 5:
        try:
            add_picture_UI = driver.find_element(by=AppiumBy.ID, value="com.niksoftware.snapseed:id/logo_view")
            add_picture_UI.click()
            time.sleep(2)
            break
        except:
            time.sleep(1)
            attempt += 1
            
    attempt = 0
    while attempt < 5:
        try:
            permission_UI = driver.find_element(by=AppiumBy.ID, value="com.android.permissioncontroller:id/permission_allow_button")
            permission_UI.click()
            time.sleep(2)
            break
        except:
            time.sleep(1)
            attempt += 1

    command = f'adb shell input keyevent 4' # back button
    result = subprocess.run(command, text=True, shell=True)
    time.sleep(1)

    command = f'adb shell input keyevent 3' # home button
    result = subprocess.run(command, text=True, shell=True)
    time.sleep(1)
    
    driver.quit()
    
    print('Snapseed app initialized successfully.')