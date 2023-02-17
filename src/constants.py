action_space = {"PreviousUnit": 1,
            "NextUnit": 2,
            "MoveLeft": 3,
            "MoveRight": 4,
            "MoveUp": 5,
            "MoveDown": 6,
            "MoveUpLeft": 7,
            "MoveUpRight": 8,
            "MoveDownLeft": 9,
            "MoveDownRight": 10,

            "Attack": 11,
            "Harvest": 12,

            "Build0": 13,
            "Build1": 14,
            "Build2": 15,
            "NoAction": 16}

inv_action_space = {v: k for k, v in action_space.items()}

