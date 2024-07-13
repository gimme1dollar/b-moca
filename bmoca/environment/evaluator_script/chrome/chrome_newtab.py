from appium.webdriver.common.appiumby import AppiumBy

def check_new_tab(driver):
    try:
        tab_UI = driver.find_element(AppiumBy.ID, 
                                     'com.android.chrome:id/tab_switcher_button')
        content_desc = tab_UI.get_attribute("content-desc")
        
        if '2' in content_desc:
            return True
        else:
            return False
    except Exception:
        return False