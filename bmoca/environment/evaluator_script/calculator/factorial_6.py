from appium.webdriver.common.appiumby import AppiumBy

def check_factorial_6(driver):
    try:
        formula_UI = driver.find_element(AppiumBy.ID, 'com.google.android.calculator:id/formula')
        formula = formula_UI.get_attribute("text")
        
        return formula == '6!'
    except Exception:
        return False