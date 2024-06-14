from appium.webdriver.common.appiumby import AppiumBy

def check_noir_s03(driver):
    try:
        s03_UI = driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR , 'new UiSelector().text("S03")')
        return s03_UI.get_attribute("selected") == 'true'
    except:
        return False