from appium.webdriver.common.appiumby import AppiumBy

def check_go_to_store_map(driver):
    try:
        instore_UI = driver.find_element(AppiumBy.ID, 'com.walmart.android:id/instoremaps_webview_container') 
        in_instore = instore_UI.get_attribute('displayed') == 'true'
        return in_instore 
    except:
        return False