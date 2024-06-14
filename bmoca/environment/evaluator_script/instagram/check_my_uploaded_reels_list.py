from appium.webdriver.common.appiumby import AppiumBy

def check_check_my_uploaded_reels_list(driver):
    try:
        profile_tab_UI = driver.find_element(AppiumBy.ID, 
                                         'com.instagram.android:id/profile_tab')
        in_profile_tab = profile_tab_UI.get_attribute("selected") == 'true'
        
        reels_section_UI = driver.find_element(AppiumBy.XPATH, 
                                         '//android.widget.HorizontalScrollView[@resource-id="com.instagram.android:id/profile_tab_layout"]/android.widget.LinearLayout/android.widget.LinearLayout[2]/android.widget.FrameLayout')
        in_reels_section = reels_section_UI.get_attribute("selected") == 'true'
        
        return in_profile_tab and in_reels_section
    except Exception:
        return False
    