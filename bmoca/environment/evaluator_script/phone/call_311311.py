from appium.webdriver.common.appiumby import AppiumBy


def check_call_311311(driver):
    try:
        end_call_UI = driver.find_element(
            AppiumBy.ID, "com.android.dialer:id/incall_end_call"
        )
        number_UI = driver.find_element(
            AppiumBy.ID, "com.android.dialer:id/contactgrid_contact_name"
        )
        number = number_UI.get_attribute("text")

        return str(number).replace(" ", "") == "311311"
    except Exception:
        return False
