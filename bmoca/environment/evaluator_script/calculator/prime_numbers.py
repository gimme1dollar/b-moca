from appium.webdriver.common.appiumby import AppiumBy

def check_5_prime_numbers(driver):
    try:
        formula_UI = driver.find_element(AppiumBy.ID, 'com.google.android.calculator:id/formula')
        formula = formula_UI.get_attribute("text")
        
        return formula == '2+3+5+7+11'
    except Exception:
        return False