from appium.webdriver.common.appiumby import AppiumBy

def check_open_calculator(driver):
    try:
        _ = driver.find_element(AppiumBy.ID, 
                                        'com.google.android.calculator:id/clr')
        return True
    except Exception:
        return False