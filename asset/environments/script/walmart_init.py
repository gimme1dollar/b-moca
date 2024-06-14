import time
import subprocess
from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy

def walmart_init(avd_name='pixel_3_train_00'):
    options = AppiumOptions()
    options.load_capabilities({
        "platformName": "Android",
        "appium:platformVersion": "10",
        "appium:deviceName": avd_name,
        "appium:automationName": "UiAutomator2",
        "appium:appPackage": "com.walmart.android",
        "appium:appActivity": "com.walmart.glass.integration.splash.SplashActivity",
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
            guest_UI = driver.find_element(by=AppiumBy.ID, value="com.walmart.android:id/onboarding_welcome_secondary_button")
            guest_UI.click()
            time.sleep(1.5)
            break
        except:
            time.sleep(1)
            attempt += 1
    
    attempt = 0
    while attempt < 5:
        try:
            share_location_UI = driver.find_element(by=AppiumBy.ID, value="com.walmart.android:id/onboarding_enable_location_button")
            share_location_UI.click()
            time.sleep(1.5)
            break
        except:
            time.sleep(1)
            attempt += 1
    
    attempt = 0
    while attempt < 5:
        try:
            permision_allow_UI = driver.find_element(by=AppiumBy.ID, value="com.android.permissioncontroller:id/permission_allow_foreground_only_button")
            permision_allow_UI.click()
            time.sleep(6)
            break
        except:
            time.sleep(1)
            attempt += 1
            
    try:
        close_UI = driver.find_element(by=AppiumBy.XPATH, value='//android.widget.Button[@content-desc="Close, My Items information"]')
        close_UI.click()
        time.sleep(0.5)
    except:
        pass
    
    command = f'adb shell input keyevent 3' # home button
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(1)
    
    driver.quit()
    
    print('Walmart app initialized successfully.')
    
    