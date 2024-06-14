from appium.webdriver.common.appiumby import AppiumBy

def check_go_to_services(driver):
    try:
        services_UI = driver.find_element(AppiumBy.ID, 'com.walmart.android:id/navigation_services') 
        in_services = services_UI.get_attribute('selected') == 'true'
        return in_services 
    except:
        return False