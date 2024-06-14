from appium.webdriver.common.appiumby import AppiumBy

def check_go_to_search(driver):
    try:
        search_UI = driver.find_element(AppiumBy.ID, 'com.walmart.android:id/navigation_search') 
        in_search = search_UI.get_attribute('selected') == 'true'
        return in_search 
    except:
        return False