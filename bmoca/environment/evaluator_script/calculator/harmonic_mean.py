from appium.webdriver.common.appiumby import AppiumBy

def check_harmonic_mean(driver):
    
    result_preview = ""
    result_final = ""
    try:
        result_preview_UI = driver.find_element(AppiumBy.ID, 'com.google.android.calculator:id/result_preview')
        result_preview = result_preview_UI.get_attribute("text")
    except Exception:
        pass
    
    try:
        result_final_UI = driver.find_element(AppiumBy.ID, 'com.google.android.calculator:id/result_final')
        result_final = result_final_UI.get_attribute("text")
    except Exception:
        pass
    
    if result_preview.startswith("4.44") or result_final.startswith("4.44"):
        return True
    else:
        return False