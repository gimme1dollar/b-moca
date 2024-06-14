from appium.webdriver.common.appiumby import AppiumBy

def check_sqrt_25(driver):
    try:
        formula_UI = driver.find_element(AppiumBy.ID, 'com.google.android.calculator:id/formula')
        formula = formula_UI.get_attribute("text")
        
        return formula == '√25'
    except Exception:
        return False