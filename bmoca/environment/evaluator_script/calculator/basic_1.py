from appium.webdriver.common.appiumby import AppiumBy

def check_basic_1(driver):
    try:
        formula_UI = driver.find_element(AppiumBy.ID, 'com.google.android.calculator:id/formula')
        formula = formula_UI.get_attribute("text")
        
        return formula == '1+1'
    except Exception:
        return False