def confirm(prompt='Yes or no?'):
    while True:
        response = input(prompt + ' ').lower()
        if response in ['yes', 'no', 'y', 'n', 'yep', 'yeah', 'nope']:
            break
        print('Please type yes or no.')
    if response in ['yes', 'y', 'yep', 'yeah']:
        return True
    return False


def integer(prompt='Enter a number:', positive=None, zero=True):
    while True:
        response = input(prompt + ' ')
        if not all(c.isdigit() for c in response) or len(response) == 0:
            print('Please enter numeric characters.')
            continue
        response = int(response)
        if positive is not None and positive and response < 0:
            print('Input must be positive.')
        elif positive is not None and not positive and response > 0:
            print('Input must be negative.')
        elif not zero and response == 0:
            print('Input must be nonzero.')
        else:
            break
    return response
