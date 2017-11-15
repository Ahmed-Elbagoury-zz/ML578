import csv
def preprocess_transcation_table(input_filename, output_filename):
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
        # Skip payment_method_id field as it is a categorical data.
        payment_plan_days = float(row[2])
        plan_list_price = float(row[3])
        actual_amount_paid = float(row[4])
        is_auto_renew = float(row[5])
        # Skip transaction_date and membership_expire_date as it is
        # categorical data.
        is_cancel = float(row[8])
        user_record_count = 1
        user_data = []
        if user_id in users_dict:
            user_data = users_dict[user_id]
            user_data[0] += payment_plan_days
            user_data[1] += plan_list_price
            user_data[2] += actual_amount_paid
            user_data[3] += is_auto_renew
            user_data[4] += is_cancel
            user_data[5] += user_record_count            
        else: # First time to encouter this user.
            user_data = [payment_plan_days, plan_list_price, actual_amount_paid,
                         is_auto_renew, is_cancel, user_record_count]           
        # Add user to the dictionary.
        users_dict[user_id] = user_data
   # Average user records and then write it to the output file.
   output_file = open(output_filename, "w")
   output_file.write('msno,payment_plan_days,plan_list_price,actual_amount_paid,is_auto_renew,is_cancel\n')
   for user_id in users_dict:
       user_data = users_dict[user_id]
       user_records_number = user_data[5]
       output_line = user_id
       fields_count = 0
       for user_attribute in user_data:
           user_attribute = user_attribute / user_records_number
           output_line = output_line + ',' + str(user_attribute)
           fields_count = fields_count + 1
           if fields_count == 5:
               # Fifth field is the count field, don't include.
               break
       output_file.write(output_line + '\n')
   output_file.close()
 
    
