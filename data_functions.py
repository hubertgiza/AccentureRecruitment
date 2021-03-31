def sex_from_name(dataFrame):
    with open("Additional_Files/polish_male_firstnames.txt") as file:
        male_names = file.read().splitlines()
    with open("Additional_Files/polish_female_firstnames.txt") as file:
        female_names = file.read().splitlines()
    for i, record in enumerate(dataFrame['sex'].values):
        if record != 'male' and record != 'female':
            if dataFrame['name'][i] in male_names:
                dataFrame['sex'][i] = 'male'
            elif dataFrame['name'][i] in female_names:
                dataFrame['sex'][i] = 'female'


def sex_to_binary(dataFrame):
    dataFrame['sex'] = dataFrame['sex'].apply(lambda x: 1 if x == 'female' else 0)


def sex_preprocessing(dataFrame):
    sex_from_name(dataFrame)
    sex_to_binary(dataFrame)


def relationship_preprocessing(dataFrame):
    dataFrame['relationship_status'] = dataFrame['relationship_status'].apply(lambda x: 0 if x == 'Single'
    else 1 if x == 'In relationship' else 2 if x == 'Married' else 3 if x == 'Married with kids'
    else 4 if x == 'Divorced' else float('NaN'))


def dob_preprocessing(dataFrame):
    dataFrame['dob'] = dataFrame['dob'].apply(lambda x: (2021 - int(x[0:4])) if isinstance(x, str) else float('NaN'))


def data_preprocessing(dataFrame):
    sex_preprocessing(dataFrame)
    relationship_preprocessing(dataFrame)
    dob_preprocessing(dataFrame)

