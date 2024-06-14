import bmoca.environment.evaluator_script.base as base_evaluator
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
        
        # base
        
        if task_instruction == 'Goal: call the white house (202-456-1111)':
            return base_evaluator.check_call_white_house(driver)
        
        elif task_instruction == 'Goal: call 311311':
            return base_evaluator.check_call_311311(driver)
        
        elif task_instruction == 'Goal: call the US national contact center (800-333-4636)':
            return base_evaluator.check_call_contact_center(driver)
        
        elif task_instruction == 'Goal: create alarm at 10:30 am on every midweek':
            return base_evaluator.check_alarm1030am_midweek(driver)

        elif task_instruction == 'Goal: create alarm at 10:30 am on every weekday':
            return base_evaluator.check_alarm1030am_weekday(driver)
        
        elif task_instruction == 'Goal: create alarm at 10:30 am on every weekend':
            return base_evaluator.check_alarm1030am_weekend(driver)
        
        elif task_instruction == 'Goal: open a new tab in Chrome':
            return base_evaluator.check_new_tab(driver)

        # Instagram
        
        elif task_instruction == "Goal: go to list of posts I'm tagged in on Instagram":
            return instagram_evaluator.check_check_list_of_posts_I_tagged(driver)
        
        elif task_instruction == 'Goal: go to my uploaded reels list in Instagram':
            return instagram_evaluator.check_check_my_uploaded_reels_list(driver)
        
        elif task_instruction == 'Goal: open Instagram':
            return instagram_evaluator.check_go_to_feed_tab(driver)
        
        elif task_instruction == 'Goal: go to reels tab in Instagram':
            return instagram_evaluator.check_go_to_reels_tab(driver)
            
        elif task_instruction == 'Goal: go to my profile in Instagram':
            return instagram_evaluator.check_go_to_my_profile (driver)
        
        elif task_instruction == 'Goal: go to search tab in Instagram':
            return instagram_evaluator.check_go_to_search_tab(driver)
        
        # Snapseed
        
        elif task_instruction == 'Goal: open image in Snapseed':
            return snapseed_evaluator.check_open_image(driver)
        
        elif task_instruction == 'Goal: open Snapseed':
            return snapseed_evaluator.check_open_snapseed(driver)
        
        elif task_instruction == 'Goal: open image and go to tools tab in Snapseed':
            return snapseed_evaluator.check_open_tools(driver)
        
        elif task_instruction == 'Goal: set dark theme in Snapseed':
            return snapseed_evaluator.check_dark_theme(driver)
        
        elif task_instruction == 'Goal: set format quality to JPG 100% in Snapseed':
            return snapseed_evaluator.check_format_quality_100(driver)
        
        elif task_instruction == 'Goal: set image sizing to 2000 px':
            return snapseed_evaluator.check_image_sizing_2000(driver)
        
        elif task_instruction == 'Goal: open image and apply noir S03 filter in Snapseeed':
            return snapseed_evaluator.check_noir_s03(driver)
        
        elif task_instruction == 'Goal: open image and apply portrait filter in Snapseed':
            return snapseed_evaluator.check_portrait_filter(driver)
        
        # Wikipedia
        
        elif task_instruction == "Goal: open Wikipedia":
            return wikipedia_evaluator.check_go_to_explore_tab(driver)
        
        elif task_instruction == "Goal: go to saved tab in Wikipedia":
            return wikipedia_evaluator.check_go_to_saved_tab(driver)
        
        elif task_instruction == "Goal: go to search tab in Wikipedia":
            return wikipedia_evaluator.check_go_to_search_tab(driver)
        
        elif task_instruction == "Goal: disable the top 2 topics in the feed customization settings on Wikipedia and go back to the feed":
            return wikipedia_evaluator.check_disable_top2(driver)
        
        elif task_instruction == "Goal: disable the top 1 and 'randomizer' topics in the feed customization settings on Wikipedia and go back to the feed":
            return wikipedia_evaluator.check_disable_top1_random(driver)
        
        elif task_instruction == "Goal: disable the top 2 and 'randomizer' topics in the feed customization settings on Wikipedia and go back to the feed":
            return wikipedia_evaluator.check_disable_top2_random(driver)
        
        elif task_instruction == "Goal: disable the topics with odd-numbered indices in the feed customization settings on Wikipedia and go back to the feed":
            return wikipedia_evaluator.check_disable_odd(driver)
        
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
        
        elif task_instruction == "Goal: go to grocery category and show subcategories in Walmart":
            return walmart_evaluator.check_go_to_grocery_category(driver)
        
        elif task_instruction == "Goal: go to my cart in Walmart":
            return walmart_evaluator.check_go_to_my_cart(driver)
        
        elif task_instruction == "Goal: go to store map in Walmart":
            return walmart_evaluator.check_go_to_store_map(driver)
        
        
        # Calculator
        
        elif task_instruction == "Goal: open Calculator":
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
        
        elif task_instruction == "Goal: input for computing factorial of 6 in Calculator":
            return calculator_evaluator.check_factorial_6(driver)
        
        elif task_instruction == "Goal: input for computing square root of 25 in Calculator":
            return calculator_evaluator.check_sqrt_25(driver)

        # else

        else:
            return False


