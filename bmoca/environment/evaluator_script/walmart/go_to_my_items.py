from appium.webdriver.common.appiumby import AppiumBy

def check_go_to_my_items(driver):
    try:
        my_items_UI = driver.find_element(AppiumBy.ID, 'com.walmart.android:id/navigation_my_items') 
        in_my_items = my_items_UI.get_attribute('selected') == 'true'
        return in_my_items
    except:
        return False