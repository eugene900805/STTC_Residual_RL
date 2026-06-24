"""Reinforcement-learning comparison for the WMR slipping trajectory tracking.

`wmr_env`   gymnasium environment wrapping the slipping WMR plant
`train`     train a SAC policy
`evaluate`  run the trained policy on the paper's scenarios and compare against
            the paper's kinematic + adaptive backstepping controller
"""
