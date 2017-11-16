import random
import os.path as path
import csv
import math
def split_data_random(file_name, split_number, output_folder, train_ratio):
    csvfile = open(file_name,'r')
    print('Load Data\n');
    lines = csvfile.readlines()
    header = lines[0]
    del lines[0]
    random.seed(0)
    random.shuffle(lines)
    line_start = 0
    lines_per_split_number = math.floor(len(lines) / split_number)
    for split in range(split_number):
        # Write training data of the split.
        output_train_file = open(path.join(output_folder,
                                           str(split) + '_train.csv'), 'w')
        output_train_file.write(header)
        line_end = lines_per_split_number + line_start
        line_end_train = math.floor(train_ratio * lines_per_split_number) + line_start
        while line_start < line_end_train:
            output_train_file.write(lines[line_start])
            line_start = line_start + 1
        output_train_file.close()
        # Write testing data of the split.
        output_test_file = open(path.join(output_folder,
                                          str(split) + '_test.csv'), 'w')
        output_test_file.write(header)
        while line_start < line_end:
            output_test_file.write(lines[line_start])
            line_start = line_start + 1
        output_test_file.close()

        

