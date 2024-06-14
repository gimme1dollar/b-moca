from appium.webdriver.common.appiumby import AppiumBy

def check_check_list_of_posts_I_tagged(driver):
    try:
        profile_tab_UI = driver.find_element(AppiumBy.ID, 
                                         'com.instagram.android:id/profile_tab')
        in_profile_tab = profile_tab_UI.get_attribute("selected") == 'true'
        
        tagged_section_UI = driver.find_element(AppiumBy.XPATH, 
                                         '//android.widget.HorizontalScrollView[@resource-id="com.instagram.android:id/profile_tab_layout"]/android.widget.LinearLayout/android.widget.LinearLayout[3]/android.widget.FrameLayout')
        in_tagged_section = tagged_section_UI.get_attribute("selected") == 'true'
        
        return in_profile_tab and in_tagged_section
    except Exception:
        return False