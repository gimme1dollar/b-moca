import time
import subprocess
from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy

def wikipedia_init(avd_name='pixel_3_train_00'):
    options = AppiumOptions()
    options.load_capabilities({
        "platformName": "Android",
        "appium:platformVersion": "10",
        "appium:deviceName": avd_name,
        "appium:automationName": "UiAutomator2",
        "appium:appPackage": "org.wikipedia",
        "appium:appActivity": "org.wikipedia.main.MainActivity",
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
            skip_UI = driver.find_element(by=AppiumBy.ID, value="org.wikipedia:id/fragment_onboarding_skip_button")
            skip_UI.click()
            time.sleep(2)
            break
        except:
            time.sleep(1)
            attempt += 1
    
    attempt = 0
    while attempt < 5:
        try:
            got_it_UI = driver.find_element(by=AppiumBy.ID, value="org.wikipedia:id/view_announcement_action_negative")
            got_it_UI.click()
            time.sleep(2)
            break
        except:
            time.sleep(1)
            attempt += 1
    
    command = f'adb shell input keyevent 3' # home button
    result = subprocess.run(command, text=True, shell=True)
    time.sleep(1)
    
    driver.quit()
    
    print('Wikipedia app initialized successfully.')
    