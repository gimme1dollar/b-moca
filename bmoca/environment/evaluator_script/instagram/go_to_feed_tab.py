from appium.webdriver.common.appiumby import AppiumBy


def check_go_to_feed_tab(driver):
    try:
        feed_icon = driver.find_element(AppiumBy.ID, 
                                        'com.instagram.android:id/feed_tab')
        in_feed_tab = feed_icon.get_attribute("selected")
        return in_feed_tab == 'true'
    except Exception:
        return False
    