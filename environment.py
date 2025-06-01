import numpy as np
import robosuite as suite
from policies import *
import time
from nlp import process_command


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
move_step = 0.05  


def execute_command(command, obs):
    """Execute the given command and return the new observation."""
    action = np.zeros(7)
    
    if command == "break":
        print("ENDING PROGRAM, THANK YOU FOR INTERACTING!")
        return obs, True  # Return True to indicate program should end
        
    elif command == "move left":
        start = time.time()
        while (time.time() - start < 3):
            action[1] = 0.05
            obs, reward, done, info = env.step(action)
            env.render()
            
    elif command == "move right":
        start = time.time()
        while time.time() - start < 3:
            action[1] = -move_step
            obs, reward, done, info = env.step(action)
            env.render()
            
    elif command == "move forward":
        start = time.time()
        while time.time() - start < 3:
            action[0] = move_step
            obs, reward, done, info = env.step(action)
            env.render()
            
    elif command == "move back":
        start = time.time()
        while time.time() - start < 3:
            action[0] = -move_step
            obs, reward, done, info = env.step(action)
            env.render()
            
    elif command == "move up":
        start = time.time()
        while time.time() - start < 3:
            action[2] = move_step
            obs, reward, done, info = env.step(action)
            env.render()
            
    elif command == "move down":
        start = time.time()
        while time.time() - start < 3:
            action[2] = -move_step
            obs, reward, done, info = env.step(action)
            env.render()
            
    elif command == "gripper open":
        start = time.time()
        while time.time() - start < 1:
            action[6] = -1
            obs, reward, done, info = env.step(action)
            env.render()
            
    elif command == "gripper close":
        start = time.time()
        while time.time() - start < 1:
            action[6] = 1
            obs, reward, done, info = env.step(action)
            env.render()
            
    elif command == "reset":
        print("RESETTING ENVIRONMENT")
        obs = env.reset()
        policy = None
        prev_policy = None
        
    elif command == "lift green":
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
        
    elif command == "lift red":
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
        
    elif command == "stack":
        print("Starting Stacking Policy!")
        policy = StackPolicy(obs)
        start = time.time()
        while (time.time() - start < 8):
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            env.render()
        print("COMPLETED STACK!")
        prev_policy = "StackPolicy"
        policy = None
        
    return obs, False


#MAIN LOOP===============================
while (True):
    action = np.zeros(7)
    obs, reward, done, info = env.step(action)
    env.render()


    if (np.random.rand() < 0.02): #Every 2.5 seconds, sample an input
        try:
            user_input = input("Enter a command: ")
            print("You typed:", user_input)
            
            # Process the command using NLP
            command = process_command(user_input)
            
            if command is None:
                print("UNRECOGNIZED COMMAND, TRY AGAIN!")
            else:
                obs, should_break = execute_command(command, obs)
                if should_break:
                    break
                    
        except EOFError:
            pass

