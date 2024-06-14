from appium.webdriver.common.appiumby import AppiumBy

def check_go_to_account(driver):
    try:
        account_UI = driver.find_element(AppiumBy.ID, 'com.walmart.android:id/navigation_account') 
        in_account = account_UI.get_attribute('selected') == 'true'
        return in_account 
    except:
        return False