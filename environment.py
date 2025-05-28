import numpy as np
import robosuite as suite
from policies import *
import time

'''
TEAM PLEASE READ THIS:
So basically, if you try to do like let's say lift green cube while you have a red one in your hand, it'll just reset the whole thing and do the other policy
or crash (idk what is the reason for this). To avoid this I hard coded a prev_policy so that if it's one of the ones that will crash the program you will
not be able to run it
'''

#CREATE ENVIRONMENT=======================
env = suite.make(
    env_name = "Stack", 
    robots = "Panda",
    has_renderer = True,
    has_offscreen_renderer = False,
    use_camera_obs = False,
)

danger_policies = ["LiftPolicyRed", "LiftPolicyGreen", "StackPolicy"]

obs = env.reset()
policy = None
prev_policy = None

#MAIN LOOP===============================
while (True):
    action = np.zeros(7)
    obs, reward, done, info = env.step(action)
    env.render()

    if (np.random.rand() < 0.02): #Every 2.5 seconds, sample an input
        try:
            user_input = input("Enter a command: ")
            print("You typed:", user_input)
            user_input = user_input.lower()
            print("lowered:", user_input)

            #THIS IS WHERE THE FUN BEGINS OLANA LMFAOAOAOAOAO IM GOING FUCKING INSANE======================================================
            if (user_input == "break"):
                print("ENDING PROGRAM, THANK YOU FOR INTERACTING!")
                break

            elif (prev_policy in danger_policies):
                print("UNALLOWED COMMAND, MAKE SURE POLICY IS SAFE FIRST")

            elif (user_input == "reset"):
                print("RESETTING ENVIRONMENT")
                obs = env.reset()
                policy = None
                prev_policy = None

            elif (user_input == "lift green"):
                print("Starting Lift Green Cube Policy!")
                policy = LiftPolicyGreen(obs)
                start = time.time()

                while (time.time() - start < 3):
                    action = policy.get_action(obs)
                    obs, reward, done, info = env.step(action)
                    env.render()

                print("COMPLETED LIFT GREEN!")
                prev_policy = "LiftPolicyGreen"
                policy = None

            elif (user_input == "lift red"):
                print("Starting Lift Red Cube Policy!")
                policy = LiftPolicyRed(obs)
                start = time.time()

                while (time.time() - start < 3):
                    action = policy.get_action(obs)
                    obs, reward, done, info = env.step(action)
                    env.render()

                print("COMPLETED LIFT RED!")
                prev_policy = "LiftPolicyRed"
                policy = None

            elif (user_input == "stack"):
                print("Starting Stacking Policy!")
                policy = StackPolicy(obs)
                start = time.time()

                while (time.time() - start < 7):
                    action = policy.get_action(obs)
                    obs, reward, done, info = env.step(action)
                    env.render()

                print("COMPLETED STACK!")
                prev_policy = "StackPolicy"
                policy = None

            else:
                print("UNRECOGNIZED COMMAND, TRY AGAIN!")


        except EOFError:
            pass