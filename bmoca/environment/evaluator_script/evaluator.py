import bmoca.environment.evaluator_script.chrome as chrome_evaluator
import bmoca.environment.evaluator_script.clock as clock_evaluator
import bmoca.environment.evaluator_script.phone as phone_evaluator
import bmoca.environment.evaluator_script.instagram as instagram_evaluator
import bmoca.environment.evaluator_script.snapseed as snapseed_evaluator
import bmoca.environment.evaluator_script.wikipedia as wikipedia_evaluator
import bmoca.environment.evaluator_script.calculator as calculator_evaluator
import bmoca.environment.evaluator_script.walmart as walmart_evaluator


class Evaluator:

    def __init__(self, task_instruction):
        self.task_instruction = task_instruction

    def success_detector(self, driver):
        task_instruction = self.task_instruction

        # Calculator

        if task_instruction == "Goal: open Calculator":
            return calculator_evaluator.check_open_calculator(driver)

        elif task_instruction == "Goal: input 1 in Calculator":
            return calculator_evaluator.check_tap_1(driver)

        elif task_instruction == "Goal: input '1+1' in Calculator":
            return calculator_evaluator.check_basic_1(driver)

        elif task_instruction == "Goal: input '3×5' in Calculator":
            return calculator_evaluator.check_basic_2(driver)

        elif task_instruction == "Goal: input '2+24÷3' in Calculator":
            return calculator_evaluator.check_basic_3(driver)

        elif task_instruction == "Goal: input '17×23' in Calculator":
            return calculator_evaluator.check_basic_4(driver)

        elif task_instruction == "Goal: input 'cos(60)' in Calculator":
            return calculator_evaluator.check_cos60(driver)

        elif task_instruction == "Goal: input 'cos(180)' in Calculator":
            return calculator_evaluator.check_cos180(driver)

        elif task_instruction == "Goal: input factorial of 6 in Calculator":
            return calculator_evaluator.check_factorial_6(driver)

        elif task_instruction == "Goal: input square root of 25 in Calculator":
            return calculator_evaluator.check_sqrt_25(driver)

        elif task_instruction == "Goal: input '5!÷(2!x3!)' in Calculator":
            return calculator_evaluator.check_5_choose_2(driver)

        elif task_instruction == "Goal: compute 50% of 28 ('50%28') in Calculator":
            return calculator_evaluator.check_50_percent_of_28(driver)

        elif (
            task_instruction
            == "Goal: compute the geometric mean of 3, 4, and 5 in Calculator"
        ):
            return calculator_evaluator.check_geometric_mean(driver)

        elif (
            task_instruction
            == "Goal: compute the harmonic mean of 4 and 5 in Calculator"
        ):
            return calculator_evaluator.check_harmonic_mean(driver)

        elif task_instruction == "Goal: input '10!÷(2!x8!)' in Calculator":
            return calculator_evaluator.check_10_choose_2(driver)

        elif task_instruction == "Goal: input '5!÷(2!x3!)' in Calculator":
            return calculator_evaluator.check_5_choose_2(driver)

        elif task_instruction == "Goal: input 'ln(1234)' in Calculator":
            return calculator_evaluator.check_log_1234(driver)

        elif (
            task_instruction
            == "Goal: input the formula for computing sum of the first 5 Fibonacci numbers in Calculator"
        ):
            return calculator_evaluator.check_5_fibonaccis(driver)

        elif (
            task_instruction
            == "Goal: input the formula for computing sum of the first 5 prime numbers in Calculator"
        ):
            return calculator_evaluator.check_5_prime_numbers(driver)

        elif (
            task_instruction
            == "Goal: input the formula for converting 45 degrees to radians ('45xπ÷180') in Calculator"
        ):
            return calculator_evaluator.convert_45_degrees_to_rad(driver)

        # Chrome

        elif task_instruction == "Goal: open a new tab in Chrome":
            return chrome_evaluator.check_new_tab(driver)

        # Clock

        elif task_instruction == "Goal: create alarm at 10:30 am on every midweek":
            return clock_evaluator.check_alarm1030am_midweek(driver)

        elif task_instruction == "Goal: create alarm at 10:30 am on every weekday":
            return clock_evaluator.check_alarm1030am_weekday(driver)

        elif task_instruction == "Goal: create alarm at 10:30 am on every weekend":
            return clock_evaluator.check_alarm1030am_weekend(driver)

        elif (
            task_instruction
            == "Goal: create alarm at 10:30 am in clock and increase alarm volume in setting"
        ):
            return clock_evaluator.check_alarm1030am_and_volume(driver)

        elif (
            task_instruction
            == "Goal: create alarm at 13:30 pm and another alarm 2 hours after it"
        ):
            return clock_evaluator.check_alarm1330pm_and_after_2(driver)

        elif (
            task_instruction
            == "Goal: create alarm at 13:30 pm and another alarm 2 hours before it"
        ):
            return clock_evaluator.check_alarm1330pm_and_before_2(driver)

        elif (
            task_instruction
            == "Goal: create alarm at 13:30 pm in clock and increase alarm volume in setting"
        ):
            return clock_evaluator.check_alarm1330pm_and_volume(driver)

        elif task_instruction == "Goal: create alarm at 13:30 pm on every weekday":
            return clock_evaluator.check_alarm1330pm_weekday(driver)

        elif (
            task_instruction
            == "Goal: create alarm at 13:30 pm on every weekday_and_increase_alarm_volume_in_setting"
        ):
            return clock_evaluator.check_alarm1330pm_weekday_and_volume(driver)

        elif task_instruction == "Goal: create alarm at 13:30 pm on every weekend":
            return clock_evaluator.check_alarm1330pm_weekend(driver)

        elif (
            task_instruction
            == "Goal: delete alarm at 9:00 am and create alamr at 10:30 am"
        ):
            return clock_evaluator.check_delete_9am_and_create_alarm1030am(driver)

        elif (
            task_instruction
            == "Goal: turn on alarm at 9:30 am in clock and increase alarm volume in setting"
        ):
            return clock_evaluator.check_alarm9am_and_volume(driver)

        # Instagram

        elif task_instruction == "Goal: go to list of posts I'm tagged in on Instagram":
            return instagram_evaluator.check_check_list_of_posts_I_tagged(driver)

        elif task_instruction == "Goal: go to my uploaded reels list in Instagram":
            return instagram_evaluator.check_check_my_uploaded_reels_list(driver)

        elif task_instruction == "Goal: open Instagram":
            return instagram_evaluator.check_go_to_feed_tab(driver)

        elif task_instruction == "Goal: go to reels tab in Instagram":
            return instagram_evaluator.check_go_to_reels_tab(driver)

        elif task_instruction == "Goal: go to my profile in Instagram":
            return instagram_evaluator.check_go_to_my_profile(driver)

        elif task_instruction == "Goal: go to search tab in Instagram":
            return instagram_evaluator.check_go_to_search_tab(driver)

        # Phone

        elif task_instruction == "Goal: call the white house (202-456-1111)":
            return phone_evaluator.check_call_white_house(driver)

        elif task_instruction == "Goal: call 311311":
            return phone_evaluator.check_call_311311(driver)

        elif (
            task_instruction
            == "Goal: call the US national contact center (800-333-4636)"
        ):
            return phone_evaluator.check_call_contact_center(driver)

        elif task_instruction == "Goal: call 11489":
            return phone_evaluator.check_call_11489(driver)

        elif task_instruction == "Goal: call 123-4578":
            return phone_evaluator.check_call_1234578(driver)

        elif task_instruction == "Goal: call 223-4458":
            return phone_evaluator.check_call_2234458(driver)

        elif task_instruction == "Goal: call 26-445-1193":
            return phone_evaluator.check_call_264451193(driver)

        elif task_instruction == "Goal: call 402-7717":
            return phone_evaluator.check_call_4027717(driver)

        elif task_instruction == "Goal: call 766-3394":
            return phone_evaluator.check_call_7663394(driver)

        elif task_instruction == "Goal: call 987-6654":
            return phone_evaluator.check_call_9876654(driver)

        elif (
            task_instruction == "Goal: call the national weather service (301-713-0622)"
        ):
            return phone_evaluator.check_call_weather(driver)

        elif (
            task_instruction
            == "Goal: call the social security administration (800-772-1213)"
        ):
            return phone_evaluator.check_call_social(driver)

        # Snapseed

        elif task_instruction == "Goal: open image in Snapseed":
            return snapseed_evaluator.check_open_image(driver)

        elif task_instruction == "Goal: open Snapseed":
            return snapseed_evaluator.check_open_snapseed(driver)

        elif task_instruction == "Goal: open image and go to tools tab in Snapseed":
            return snapseed_evaluator.check_open_tools(driver)

        elif task_instruction == "Goal: set dark theme in Snapseed":
            return snapseed_evaluator.check_dark_theme(driver)

        elif task_instruction == "Goal: set format quality to JPG 100% in Snapseed":
            return snapseed_evaluator.check_format_quality_100(driver)

        elif task_instruction == "Goal: set image sizing to 2000 px":
            return snapseed_evaluator.check_image_sizing_2000(driver)

        elif (
            task_instruction
            == "Goal: open image and apply noir S03 filter in Snapseeed"
        ):
            return snapseed_evaluator.check_noir_s03(driver)

        elif (
            task_instruction == "Goal: open image and apply portrait filter in Snapseed"
        ):
            return snapseed_evaluator.check_portrait_filter(driver)

        elif (
            task_instruction
            == "Goal: apply noir S03 filter to an image after setting format quality to JPG 100% in Snapseed"
        ):
            return snapseed_evaluator.check_apply_filter_after_quality(driver)

        elif (
            task_instruction
            == "Goal: apply noir S03 filter to an image after setting image sizing to 2000 px in Snapseed"
        ):
            return snapseed_evaluator.check_appy_filter_after_sizing(driver)

        elif (
            task_instruction
            == "Goal: apply noir S03 filter to an image after setting dark theme in Snapseed"
        ):
            return snapseed_evaluator.check_apply_filter_after_theme(driver)

        # Wikipedia

        elif task_instruction == "Goal: open Wikipedia":
            return wikipedia_evaluator.check_go_to_explore_tab(driver)

        elif task_instruction == "Goal: go to saved tab in Wikipedia":
            return wikipedia_evaluator.check_go_to_saved_tab(driver)

        elif task_instruction == "Goal: go to search tab in Wikipedia":
            return wikipedia_evaluator.check_go_to_search_tab(driver)

        elif (
            task_instruction
            == "Goal: disable the top 2 topics in the feed customization settings on Wikipedia and go back to the feed"
        ):
            return wikipedia_evaluator.check_disable_top2(driver)

        elif (
            task_instruction
            == "Goal: disable the top 1 and 'randomizer' topics in the feed customization settings on Wikipedia and go back to the feed"
        ):
            return wikipedia_evaluator.check_disable_top1_random(driver)

        elif (
            task_instruction
            == "Goal: disable the top 2 and 'randomizer' topics in the feed customization settings on Wikipedia and go back to the feed"
        ):
            return wikipedia_evaluator.check_disable_top2_random(driver)

        elif (
            task_instruction
            == "Goal: disable the topics with odd-numbered indices in the feed customization settings on Wikipedia and go back to the feed"
        ):
            return wikipedia_evaluator.check_disable_odd(driver)

        elif task_instruction == "Goal: decrease the text size to 50% in Wikipedia":
            return wikipedia_evaluator.check_text_size_50(driver)

        elif task_instruction == "Goal: increase the text size to 180% in Wikipedia":
            return wikipedia_evaluator.check_text_size_180(driver)

        elif (
            task_instruction
            == "Goal: disable featured article feed, decrease the text size to 50%, and return to the feed on Wikipedia"
        ):
            return wikipedia_evaluator.check_disable_and_text_size_50(driver)

        elif (
            task_instruction
            == "Goal: disable featured article feed, increase the text size to 180%, and return to the feed on Wikipedia"
        ):
            return wikipedia_evaluator.check_disable_and_text_size_180(driver)

        elif (
            task_instruction
            == "Goal: disable the 'show link previews', 'top read' feed settings, and return to the feed on Wikipedia"
        ):
            return wikipedia_evaluator.check_disable_preview_and_feed(driver)

        elif (
            task_instruction
            == "Goal: disable the topics that are related to 'history' in the feed customization settings on Wikipedia and go back to the feed"
        ):
            return wikipedia_evaluator.check_disable_history_topics(driver)

        elif (
            task_instruction
            == "Goal: disable the topics that include 'day' in their names in the feed customization settings on Wikipedia and return to the feed"
        ):
            return wikipedia_evaluator.check_disable_day_topics(driver)

        elif (
            task_instruction
            == "Goal: disable the topics with even-numbered indices in the feed customization settings on Wikipedia and go back to the feed"
        ):
            return wikipedia_evaluator.check_disable_even(driver)

        elif (
            task_instruction
            == "Goal: disable the topics with prime-numbered indices in the feed customization settings on Wikipedia and go back to the feed"
        ):
            return wikipedia_evaluator.check_disable_prime(driver)

        # Walmart

        elif task_instruction == "Goal: open Walmart":
            return walmart_evaluator.check_open_walmart(driver)

        elif task_instruction == "Goal: go to my items tab in Walmart":
            return walmart_evaluator.check_go_to_my_items(driver)

        elif task_instruction == "Goal: go to search tab in Walmart":
            return walmart_evaluator.check_go_to_search(driver)

        elif task_instruction == "Goal: go to services tab in Walmart":
            return walmart_evaluator.check_go_to_services(driver)

        elif task_instruction == "Goal: go to account tab in Walmart":
            return walmart_evaluator.check_go_to_account(driver)

        elif (
            task_instruction
            == "Goal: go to grocery category and show subcategories in Walmart"
        ):
            return walmart_evaluator.check_go_to_grocery_category(driver)

        elif task_instruction == "Goal: go to my cart in Walmart":
            return walmart_evaluator.check_go_to_my_cart(driver)

        elif task_instruction == "Goal: go to store map in Walmart":
            return walmart_evaluator.check_go_to_store_map(driver)

        # else

        else:
            return False
