from appium.webdriver.common.appiumby import AppiumBy

def check_go_to_grocery_category(driver):
    try:
        category_title_UI = driver.find_element(AppiumBy.ID, 'com.walmart.android:id/category_container_title') 
        in_grocery_catagory = category_title_UI.get_attribute('text') == 'Grocery'
        return in_grocery_catagory 
    except:
        return False