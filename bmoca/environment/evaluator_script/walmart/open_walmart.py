from appium.webdriver.common.appiumby import AppiumBy

def check_open_walmart(driver):
    try:
        home_UI = driver.find_element(AppiumBy.ID, 'com.walmart.android:id/navigation_shop') 
        return True
    except:
        return False