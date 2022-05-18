
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

def model_choice():
    """
    Function to get model choice from user
    """
    model_selected = False
    while not model_selected:
        print("Select a model:")
        models = ["UAV_gauss", "UAV_dryden", "1room heating"]
        for i, model in enumerate(models):
            print(str(i) + "- "+model)
        model_sel = input()
        model_selected=True
        try:
            model_int = int(model_sel)
        except(ValueError):
            model_selected=False
            print("Please input a numerical value matching the chosen model")
        try:
            model = models[model_int]
        except(IndexError):
            model_selected=False
            print("Please select a model number between 0 and "+str(len(models)-1))
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
