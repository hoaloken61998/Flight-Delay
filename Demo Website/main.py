from flask import Flask,render_template,request
import pickle
app=Flask(__name__)

@app.route('/',methods=['GET'])
def HomePage():
 return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def index():
 if request.method=='POST':
  try:
   dep_delay = float(request.form['DEP_DELAY'])
   crs_elapsed_time = float(request.form['CRS_ELAPSED_TIME'])
   distance = float(request.form['DISTANCE'])
   quarter_encoded = float(request.form['QUARTER'])
   month_encoded = float(request.form['MONTH'])
   day_of_month_encoded = float(request.form['DAY_OF_MONTH'])
   day_of_week_encoded = float(request.form['DAY_OF_WEEK'])
   op_unique_carrier_encoded = float(request.form['OP_UNIQUE_CARRIER'])
   tail_num_encoded = float(request.form['TAIL_NUM'])
   op_carrier_fl_num_encoded = float(request.form['OP_CARRIER_FL_NUM'])
   origin_airport_id_encoded = float(request.form['ORIGIN_AIRPORT_ID'])
   origin_city_market_id_encoded = float(request.form['ORIGIN_CITY_MARKET_ID'])
   dest_airport_id_encoded = float(request.form['DEST_AIRPORT_ID'])
   dest_city_market_id_encoded = float(request.form['DEST_CITY_MARKET_ID'])
   crs_dep_time_encoded = float(request.form['CRS_DEP_TIME'])
   crs_arr_time_encoded = float(request.form['CRS_ARR_TIME'])

   print('HI')
   filename = r'D:\COMPETITION\NCKH\CODE WEBSITE\Flight_Delay\final_model.pickle'
   loaded_model = pickle.load(open(filename, 'rb'))

   import numpy as np
   prediction=loaded_model.predict([[dep_delay, crs_elapsed_time, distance, quarter_encoded, month_encoded, day_of_month_encoded, day_of_week_encoded, op_unique_carrier_encoded, tail_num_encoded, op_carrier_fl_num_encoded, origin_airport_id_encoded, origin_city_market_id_encoded, dest_airport_id_encoded, dest_city_market_id_encoded, crs_dep_time_encoded, crs_arr_time_encoded ]])
   for i in prediction:
    if i==1:
     prediction='will be'
    else:
     prediction='wont get'

   return render_template('home.html',prediction=prediction)
  except Exception as e:
    print('The Exception message is: ',e)
    return 'something is wrong'
 else:
  return render_template('home.html'),

if __name__ == "__main__":
    app.run()

# -0.527982, -1.035133, -0.984516, -0.471838, -0.164130, 0.363424, -1.497886, 0.542972, 0.487690, 1.529449, -0.414181, 0.340695, 0.601933, 0.088269, -0.627608, -0.300743