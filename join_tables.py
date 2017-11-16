import csv
def join_tables(preprocessed_transcation_file, preprocessed_user_logs_file,
                train_file, members_file, output_filename):
    users_dict = {};
    # Load users from the train file.
    train_csvfile = open(train_file,'rb')
    is_header = True
    print('Load Train Data\n');
    row_count = 0
    for row in csv.reader(train_csvfile, delimiter = ','):
        if is_header: # Skip header line.
            is_header = False
            continue
        row_count = row_count+1
        if row_count % 1000000 == 0:
            print('In load train, row count = ' + str(row_count) + '\n')
        user_id = row[0]
        is_churn = row[1]
        users_dict[user_id] = [is_churn]
    # Load users attributes from members file.
    members_csvfile = open(members_file,'rb')
    is_header = True
    print('Load Members Data\n');
    row_count = 0
    for row in csv.reader(members_csvfile, delimiter = ','):
        if is_header: # Skip header line.
            is_header = False
            continue
        row_count = row_count+1
        if row_count % 1000000 == 0:
            print('In load member, row count = ' + str(row_count) + '\n')
        user_id = row[0]
        age = row[2]
        gender = row[3]
        if gender == 'female':
            gender = 0
        elif gender == 'male':
            gender = 1
        else: # Missing field, skip.
            continue
        if int(age) < 0: # Invalid age value.
            continue
        if user_id in users_dict:
            user_data = users_dict[user_id]
            user_data.append(age)
            user_data.append(gender)
            users_dict[user_id] = user_data
    # Load users attributed from transcation file.
    transcation_csvfile = open(preprocessed_transcation_file,'rb')
    is_header = True
    print('Load Transcation Data\n');
    row_count = 0
    for row in csv.reader(transcation_csvfile, delimiter = ','):
        if is_header: # Skip header line.
            is_header = False
            continue
        row_count = row_count+1
        if row_count % 1000000 == 0:
            print('In load transcations, row count = ' + str(row_count) + '\n')
        user_id = row[0]
        payment_plan_days = row[1]
        plan_list_price = row[2]
        actual_amount_paid = row[3]
        is_auto_renew = row[4]
        is_cancel = row[5]
        if user_id in users_dict:
            user_data = users_dict[user_id]
            user_data.append(payment_plan_days)
            user_data.append(plan_list_price)
            user_data.append(actual_amount_paid)
            user_data.append(is_auto_renew)
            user_data.append(is_cancel)
            users_dict[user_id] = user_data
    # Load users attributed from users log file.
    user_logs_csvfile = open(preprocessed_user_logs_file,'rb')
    is_header = True
    print('Load User Logs Data\n');
    row_count = 0
    for row in csv.reader(user_logs_csvfile, delimiter = ','):
        if is_header: # Skip header line.
            is_header = False
            continue
        row_count = row_count+1
        if row_count % 1000000 == 0:
            print('In load user logs, row count = ' + str(row_count) + '\n')
        user_id = row[0]
        num_25 = row[1]
        num_50 = row[2]
        num_75 = row[3]
        num_985 = row[4]
        num_100 = row[5]
        num_unq = row[6]
        total_secs = row[7]
        if user_id in users_dict:
            user_data = users_dict[user_id]
            user_data.append(num_25)
            user_data.append(num_50)
            user_data.append(num_75)
            user_data.append(num_985)
            user_data.append(num_100)
            user_data.append(num_unq)
            user_data.append(total_secs)
            users_dict[user_id] = user_data
    # Write output files.
    output_file = open(output_filename, "w")
    output_file.write('msno,is_churn,age,gender,payment_plan_days,'
                      +'plan_list_price,actual_amount_paid,is_auto_renew,'
                      +'is_cancel,num_25,num_50,num_75,num_985,num_100,'
                      + 'num_unq,total_hrs\n')
    for user_id in users_dict:
        user_data = users_dict[user_id]
        output_line = user_id
        if len(user_data) < 15: # contains missing atrributes.
            continue
        for user_attribute in user_data:
            output_line = output_line + ',' + str(user_attribute)
        output_file.write(output_line + '\n')
    output_file.close()
