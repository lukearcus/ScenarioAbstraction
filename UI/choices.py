
def noise_choice():
    """
    Choice of noise level, high, medium or low
    """
    noise_levels = ["Low", "Medium", "High"]
    selected = False
    while not selected:
        selected = True
        print("Select a noise level:")
        for i, noise in enumerate(noise_levels):
            print(str(i) + "- "+noise)
        noise_sel = input()
        try:
            noise_int = int(noise_sel)
        except(ValueError):
            selected=False
            print("Please input a numerical value matching the chosen noise level")
        try:
            noise = noise_levels[noise_int]
        except(IndexError):
            selected=False
            print("Please select a number between 0 and "+str(len(noise_levels)-1))
    return noise_int+1

def rooms_choice():
    """
    Function to get number of rooms from user
    """
    rooms_selected = False
    while not rooms_selected:
        print("Type a number of rooms between 1 and 10")
        rooms_sel = input()
        try:
            rooms_int = int(rooms_sel)
            if rooms_int > 0 and rooms_int < 11:
                return rooms_int
        except(ValueError):
            print("Please select a number between 1 and 10")

def model_choice():
    """
    Function to get model choice from user
    """
    model_selected = False
    while not model_selected:
        print("Select a model:")
        models = ["UAV_gauss",
                  "UAV_dryden",
                  "UAV_speed_choice",
                  "UAV_dir_choice",
                  "UAV_var_noise",
                  "1room heating",
                  "n_room_heating",
                  "steered_n_room_heating",
                  "steered_test",
                  "unsteered_test"
                 ]
        for i, model in enumerate(models):
            print(str(i) + "- "+model)
        model_sel = input()
        model_selected=True
        try:
            model_int = int(model_sel)
            try:
                model = models[model_int]
            except(IndexError):
                model_selected=False
                print("Please select a model number between 0 and "+str(len(models)-1))
        except(ValueError):
            model_selected=False
            print("Please input a numerical value matching the chosen model")
    return model

def load_choice():
    """
    Offers user option to load existing iMDP abstraction
    """
    while True:
        print("Load exisiting iMDP abstraction? (Y/N)")
        choice = str.upper(input())
        if choice != 'Y' and choice != 'N':
            print("Please type either 'Y' or 'N'")
        else:
            return choice

def save_choice():
    while True:
        print("Save generated iMDP abstraction? (Y/N)")
        save_choice = str.upper(input())
        if save_choice != 'Y' and save_choice != 'N':
            print("Please type either 'Y' or 'N'")
        else:
            return save_choice
