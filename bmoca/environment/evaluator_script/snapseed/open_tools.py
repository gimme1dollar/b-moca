from appium.webdriver.common.appiumby import AppiumBy

def check_open_tools(driver):
    try:
        tools_UI = driver.find_element(AppiumBy.ID, 'com.niksoftware.snapseed:id/tools_button') 
        return tools_UI.get_attribute("selected") == 'true'
    except:
        return False