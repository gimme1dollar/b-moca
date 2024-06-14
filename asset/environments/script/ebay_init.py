import time
import subprocess
from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy

def ebay_init(avd_name='pixel_3_train_00'):
    options = AppiumOptions()
    options.load_capabilities({
        "platformName": "Android",
        "appium:platformVersion": "10",
        "appium:deviceName": avd_name,
        "appium:automationName": "UiAutomator2",
        "appium:appPackage": "com.ebay.mobile",
        "appium:appActivity": "com.ebay.mobile.home.impl.main.MainActivity",
        "appium:ensureWebviewsHavePages": True,
        "appium:nativeWebScreenshot": True,
        "appium:newCommandTimeout": 3600,
        "appium:connectHardwareKeyboard": True
    })
    driver = webdriver.Remote("http://127.0.0.1:4723", options=options)
    time.sleep(2)
    
    close_UI = driver.find_element(by=AppiumBy.ACCESSIBILITY_ID, value="Close")
    close_UI.click()
    time.sleep(2)

    command = f'adb shell input keyevent 3' # home button
    result = subprocess.run(command, text=True, shell=True)
    time.sleep(1)

    driver.quit()
    
    