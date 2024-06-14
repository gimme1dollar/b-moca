from appium.webdriver.common.appiumby import AppiumBy

def check_go_to_explore_tab(driver):
    try:
        explore_tab_UI = driver.find_element(AppiumBy.ID, 
                                        'org.wikipedia:id/nav_tab_explore')
        in_explore_tab = explore_tab_UI.get_attribute("selected")
        return in_explore_tab == 'true'
    except Exception:
        return False