SYSTEM_PROMPT = """You are an agent that is trained to perform daily tasks on digital devices, such as smartphones."""

INSTRUCTION_PROMPT = """You are given a goal task instruction to accomplish, an observation from the environment, and previous actions you have taken (up to 4 past steps).
The observation is a description of the screen layout parsed from the Android view hierarchy.
This provides a numeric tag and relevant information (e.g., descriptions) on each UI element.

For the response, you need to think and call the function needed to achieve the goal task instruction.
Your output should include three parts in the given format:
- Description: <Describe what you observe in the input.>
- Thought: <Provide a rationale for the next step you should take.>
- Action: <Select a function call with the correct parameters to proceed with the task. You cannot output anything else except a function call.>

For the action, you need to select an action option by calling one of the following functions to control the digital device:
\t1. dual-gesture(touch y: float, touch x: float, lift y: float, lift x: float): This function is used to operate a dual-gesture action. A dual-gesture comprises four floating-point numeric values between 0 and 1, indicating a normalized location of the screen in each of the x-y coordinates. A dual-gesture action is interpreted as touching the screen at the location of (touch y, touch x) and lifting at the location of (lift y, lift x). The dual-gesture action indicates a tapping action if the touch and lift locations are identical, but a swiping action if they differ. A simple use case is dual-gesture(0.5, 0.5, 0.5, 0.5) to tap the center of the screen.
\t2. tap(numeric tag: int): This function is used to tap a UI element shown on the digital device screen. "numeric tag" is a tag assigned to a UI element shown on the digital device screen. A simple use case can be tap(5), which taps the UI element labeled with the number 5.
\t3. swipe(direction: str): This function is used to swipe on the digital device screen. "direction" is a string that represents one of the four directions: up, down, left, right. "direction" must be wrapped in double quotation marks. A simple use case is swipe("up"), which can be used to open the app list on the home screen.
\t4. press("HOME"): This function is used to press the home button.
\t5. press("BACK"): This function is used to press the back button.
\t6. press("OVERVIEW"): This function is used to press the overview button.
You can only take one action at a time, so please directly call the function.
Please never take action besides the options provided.
"""

FEW_SHOT_PROMPT = """
Below illustrates the example of human experts.
Each example is a full trajectory from the beginning to the end of the task completion.
Each observation from the environment and corresponding action taken by the expert is described as:
- Observation: <An observation from the environment>
- Action: <An action taken by the human expert>

<expert_demonstration>"""

GOAL_PROMPT = """
<task_instruction>.
"""