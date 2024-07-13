from appium.webdriver.common.appiumby import AppiumBy

def convert_45_degrees_to_rad(driver):
    
    try:
        formula_UI = driver.find_element(AppiumBy.ID, 'com.google.android.calculator:id/formula')
        formula = formula_UI.get_attribute("text")
        
        return formula == '45×π÷180'
    except Exception:
        return False