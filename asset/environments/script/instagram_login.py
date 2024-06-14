import time
import subprocess
from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy

# For W3C actions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions import interaction
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput

def login_instagram(avd_name='pixel_3_train_00',
                    account_id='',
                    account_pw=''):
	options = AppiumOptions()
	options.load_capabilities({
		"platformName": "Android",
		"appium:platformVersion": "10",
		"appium:deviceName": avd_name,
		"appium:automationName": "UiAutomator2",
		"appium:appPackage": "com.instagram.android",
		"appium:appActivity": "com.instagram.mainactivity.LauncherActivity",
		"appium:ensureWebviewsHavePages": True,
		"appium:nativeWebScreenshot": True,
		"appium:newCommandTimeout": 3600,
		"appium:connectHardwareKeyboard": True
	})

	driver = webdriver.Remote("http://127.0.0.1:4723", options=options)
	time.sleep(4)
 
	# el6 = driver.find_element(by=AppiumBy.ACCESSIBILITY_ID, value="Username, email or mobile number")
	# el6.click()
	el7 = driver.find_element(by=AppiumBy.ANDROID_UIAUTOMATOR, value="new UiSelector().className(\"android.widget.EditText\").instance(0)")
	el7.send_keys(account_id)

	time.sleep(0.5)
	# el8 = driver.find_element(by=AppiumBy.ACCESSIBILITY_ID, value="Password")
	# el8.click()
	el9 = driver.find_element(by=AppiumBy.ANDROID_UIAUTOMATOR, value="new UiSelector().className(\"android.widget.EditText\").instance(1)")
	el9.send_keys(account_pw)

	time.sleep(0.5)
	# driver.execute_script('mobile: hideKeyboard')
	# time.sleep(0.5)
	el11 = driver.find_element(by=AppiumBy.ANDROID_UIAUTOMATOR, value="new UiSelector().text(\"Log in\")")
	el11.click()
	time.sleep(5)
 
	try:
		save_UI = driver.find_element(by=AppiumBy.ANDROID_UIAUTOMATOR, value="new UiSelector().text(\"Save\")")
		save_UI.click()
	except:
		pass

	time.sleep(3)
	command = f'adb shell input keyevent 3'
	result = subprocess.run(command, text=True, shell=True)
	driver.quit()
