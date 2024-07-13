from appium.webdriver.common.appiumby import AppiumBy

def check_10_choose_2(driver):
    
    try:
        formula_UI = driver.find_element(AppiumBy.ID, 'com.google.android.calculator:id/formula')
        formula = formula_UI.get_attribute("text")
        
        return formula == '10!÷(2!×8!)'
    except Exception:
        return False