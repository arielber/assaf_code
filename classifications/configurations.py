import pathes
import feature_calculations.configurations as cfg

header_size = cfg.header_size
num_of_features = 8657
num_of_participants = pathes.num_of_subjects
cluster_size = [2,4,6,8]
k_list = [5,7,10,15,20,25]

k_validation = 10

random_seed = 42
np_seed = 42

class_threshold = {'cv': 15, 'lto': 10}

participants_range = range(1, num_of_participants + 1)
fake_participants_range = range(200, 225)

restrictions = [10, 6, 6, 6, 6, 6, 6]


# test structure: {'name': test name, 'filter': (col number, [chosen values]), 'labeler' : (col number, map value to label), 'validation': validation technique, 'unconfound' : deprecated}
tests_pilot_configurations = [{'name': 'all alterations', 'filter' :(1, [0, 1,2,3,4]), 'labeler': (1, {0:0, 1:1, 2:1, 3:1, 4:1}), 'validation':'cv', 'unconfound' : False},
                              {'name': 'all alterations', 'filter' :(1, [0,3,4]), 'labeler': (1, {0:0, 3:1, 4:1}), 'validation':'cv', 'unconfound' : False}
                     ]

test_random_names = ['id'] + [x['name'] for x in tests_pilot_configurations]


objective_tests = [{'name': 'all manipulation', 'filter' :(1, [0,1,2,3]), 'labeler': (1, {0:0, 1:1, 2:1, 3:1}), 'validation':'cv', 'unconfound' : False},
                    {'name': '50ms manipulation', 'filter' :(1, [0,1]), 'labeler': (1, {0:0, 1:1}), 'validation':'cv', 'unconfound' : False},
                    {'name': '100ms manipulation', 'filter' :(1, [0,2]), 'labeler': (1, {0:0, 2:1}), 'validation':'cv', 'unconfound' : False},
                    {'name': '150ms manipulation', 'filter' :(1, [0,3]), 'labeler': (1, {0:0, 3:1}), 'validation':'cv', 'unconfound' : False},
                    {'name': 'unconcious manipulation', 'filter' :([1,2], [(0,1), (1,1)]), 'labeler': (1, {0:0, 1:1}), 'validation':'lto' , 'unconfound': False},]

subjective_tests = [{'name': 'all agency', 'filter' :(1, [0,1,2,3]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : False},
                    {'name': '50ms agency', 'filter' :(1, [0,1]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : False},
                    {'name': '100ms agency', 'filter' :(1, [0,2]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : False},
                    {'name': '150ms agency', 'filter' :(1, [0,3]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : False},]






