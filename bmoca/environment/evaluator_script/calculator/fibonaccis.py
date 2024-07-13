from appium.webdriver.common.appiumby import AppiumBy


def check_5_fibonaccis(driver):
    try:
        formula_UI = driver.find_element(
            AppiumBy.ID, "com.google.android.calculator:id/formula"
        )
        formula = formula_UI.get_attribute("text")

        return formula == "0+1+1+2+3" or formula == "1+1+2+3+5"
    except Exception:
        return False
