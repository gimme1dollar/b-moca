from appium.webdriver.common.appiumby import AppiumBy

def check_open_snapseed(driver):
    try:
        open_UI = driver.find_element(AppiumBy.ID, 'com.niksoftware.snapseed:id/logo_view') 
        return True
    except:
        return False