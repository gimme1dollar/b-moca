from appium.webdriver.common.appiumby import AppiumBy

def check_open_image(driver):
    try:
        looks_UI = driver.find_element(AppiumBy.ID, 'com.niksoftware.snapseed:id/looks_button') 
        return looks_UI.get_attribute("selected") == 'true'
    except:
        return False