from appium.webdriver.common.appiumby import AppiumBy

def check_go_to_search_tab(driver):
    try:
        search_tab_UI = driver.find_element(AppiumBy.ID, 
                                        'com.instagram.android:id/search_tab')
        in_search_tab = search_tab_UI.get_attribute("selected")
        return in_search_tab == 'true'
    except Exception:
        return False