from appium.webdriver.common.appiumby import AppiumBy

def check_portrait_filter(driver):
    try:
        portrait_UI = driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR , 
                                          'new UiSelector().text("Portrait")')
        return portrait_UI.get_attribute("selected") == 'true'
    except:
        return False