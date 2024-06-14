from appium.webdriver.common.appiumby import AppiumBy

def check_cos60(driver):
    try:
        formula_UI = driver.find_element(AppiumBy.ID, 'com.google.android.calculator:id/formula')
        formula = formula_UI.get_attribute("text")
        
        return formula == 'c60'
    except Exception:
        return False