import pandas as pd
import json

COLUMNS_TO_NORMALIZE = ['dob', 'daily_commute', 'friends_number', 'relationship_status', 'education',
                        'location_population']


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


def get_fitness_related_words():
    with open('Additional_Files/related_words.txt') as file:
        set_of_words = set(file.read().splitlines())
    return set_of_words


def check_if_fitness_related(data):
    if data == 0:
        return 0
    words = get_fitness_related_words()
    for word in data.split():
        if word in words:
            return 1
    return 0


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


def hobbies_preprocessing(dataFrame):
    dataFrame['hobbies'] = dataFrame['hobbies'].fillna(0)
    dataFrame['hobbies'] = dataFrame['hobbies'].apply(lambda x: x.replace(',', ' ').lower() if x != 0 else 0)
    dataFrame['hobbies'] = dataFrame['hobbies'].apply(check_if_fitness_related)


def normalize_column(dataFrame, column_name):
    min_value = dataFrame[column_name].min()
    max_value = dataFrame[column_name].max()

    def normalize_function(x, min=min_value, max=max_value):
        return (x - min) / (max - min)

    dataFrame[column_name] = dataFrame[column_name].apply(normalize_function)


def data_preprocessing(dataFrame):
    sex_preprocessing(dataFrame)
    relationship_preprocessing(dataFrame)
    dob_preprocessing(dataFrame)
    hobbies_preprocessing(dataFrame)


def data_normalization(dataFrame):
    for name in COLUMNS_TO_NORMALIZE:
        normalize_column(dataFrame, name)


def one_hot_encoding(n_classes, Y):
    results = []
    for i in range(Y.size):
        tmp = [0] * n_classes
        tmp[int(Y[i])] = 1
        results.append(tmp)
    return pd.DataFrame(results)


def get_groups_from_json(name):
    with open('Case_Assignment/' + name + '.json') as f:
        related_words = get_fitness_related_words()
        data = json.load(f)
        data = data['data']
        users_groups = []
        for record in data:
            id = record['id']
            groups = [group['group_name'] for group in record['groups']['data']]
            users_groups.append((int(id), [name.replace(',', ' ').split() for name in groups]))
        users_groups.sort(key=lambda x: x[0])
        for i in range(len(users_groups)):
            for group in users_groups[i][1]:
                for word in group:
                    if word in related_words:
                        users_groups[i] = (users_groups[i][0], 1)
                        break
                if users_groups[i][1] == 1:
                    break
            if users_groups[i][1] != 1:
                users_groups[i] = (users_groups[i][0], 0)
        dataFrame = pd.DataFrame({'groups': [value[1] for value in users_groups]})
    return dataFrame
