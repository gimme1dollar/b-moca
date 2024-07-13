from appium.webdriver.common.appiumby import AppiumBy

def check_5_choose_2(driver):

    try:
        formula_UI = driver.find_element(AppiumBy.ID, 'com.google.android.calculator:id/formula')
        formula = formula_UI.get_attribute("text")
        
        return formula == '5!÷(2!×3!)'
    except Exception:
        return False
    