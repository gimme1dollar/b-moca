from appium.webdriver.common.appiumby import AppiumBy

def check_go_to_reels_tab(driver):
    try:
        reels_icon = driver.find_element(AppiumBy.ID, 
                                         'com.instagram.android:id/clips_tab')
        in_reels_tab = reels_icon.get_attribute("selected")
        return in_reels_tab == 'true'
    except Exception:
        return False