from appium.webdriver.common.appiumby import AppiumBy

def check_go_to_my_cart(driver):
    try:
        cart_UI = driver.find_element(AppiumBy.ID, 'com.walmart.android:id/cart_fragment_constraint_layout') 
        in_cart = cart_UI.get_attribute('displayed') == 'true'
        return in_cart 
    except:
        return False