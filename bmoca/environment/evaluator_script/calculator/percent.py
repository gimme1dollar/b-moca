from appium.webdriver.common.appiumby import AppiumBy


def check_50_percent_of_28(driver):

    try:
        formula_UI = driver.find_element(
            AppiumBy.ID, "com.google.android.calculator:id/formula"
        )
        formula = formula_UI.get_attribute("text")

        return formula == "50%28"
    except Exception:
        return False
