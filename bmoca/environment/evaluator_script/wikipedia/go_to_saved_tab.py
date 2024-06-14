from appium.webdriver.common.appiumby import AppiumBy

def check_go_to_saved_tab(driver):
    try:
        saved_tab_UI = driver.find_element(AppiumBy.ID, 
                                        'org.wikipedia:id/nav_tab_reading_lists')
        in_saved_tab = saved_tab_UI.get_attribute("selected")
        return in_saved_tab == 'true'
    except Exception:
        return False