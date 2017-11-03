import csv
def preprocess_user_logs_table(input_filename, output_filename):
   users_dict = {}; 
   csvfile = open(input_filename,'rb')
   is_header = True
   row_count = 0
   for row in csv.reader(csvfile, delimiter = ','):
        if is_header: # Skip header line.
            is_header = False
            continue
        if row_count % 1000000 == 0:
            print(row_count)
        row_count = row_count + 1
        user_id = row[0]
        # Skip date field as it is a categorical data.
        num_25 = float(row[2])
        num_50 = float(row[3])
        num_75 = float(row[4])
        num_985 = float(row[5])
        num_100 = float(row[6])
        num_unq = float(row[7])
        total_hrs = float(row[8]) / 3600.0
        # Skip corrupted logs as these features can't be neg.
        if float(row[2]) < 0 or float(row[3]) < 0 or float(row[4])< 0 or float(row[5]) < 0 or float(row[6]) < 0 or float(row[7]) < 0 or float(row[8]) < 0:
            continue
        user_record_count = 1
        user_data = []
        if user_id in users_dict:
            user_data = users_dict[user_id]
            user_data[0] += num_25
            user_data[1] += num_50
            user_data[2] += num_75
            user_data[3] += num_985
            user_data[4] += num_100
            user_data[5] += num_unq
            user_data[6] += total_hrs
            user_data[7] += user_record_count            
        else: # First time to encouter this user.
            user_data = [num_25, num_50, num_75, num_985,
                         num_100, num_unq, total_hrs, user_record_count]           
        # Add user to the dictionary.
        users_dict[user_id] = user_data
   # Average user records and then write it to the output file.
   output_file = open(output_filename, "w")
   output_file.write('msno,num_25,num_50,num_75,num_985,num_100,num_unq,total_hrs\n')
   for user_id in users_dict:
       user_data = users_dict[user_id]
       user_records_number = user_data[7]
       output_line = user_id
       fields_count = 0
       for user_attribute in user_data:
           user_attribute = user_attribute / user_records_number
           output_line = output_line + ',' + str(user_attribute)
           fields_count = fields_count + 1
           if fields_count == 7:
               # Seventh field is the count field, don't include.
               break
       output_file.write(output_line + '\n')
   output_file.close()
 
    
