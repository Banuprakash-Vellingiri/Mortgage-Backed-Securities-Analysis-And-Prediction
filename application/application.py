#Prepayment Risk Predictor Application (Using Flask)
#-----------------------------------------------------------------------------
#import dependencies
from flask import Flask, render_template, request
from ppr_pipeline import prepayment_risk_prediction
#-----------------------------------------------------------------------------
application = Flask(__name__)
application .config['TEMPLATES_AUTO_RELOAD'] = True
#-----------------------------------------------------------------------------
predictor = prepayment_risk_prediction()
#-----------------------------------------------------------------------------
@application .route('/')
def index():
    return  render_template('index.html')

@application .route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        credit_score = float(request.form['credit_score'])
        dti = float(request.form['dti'])
        months_in_repayment = int(request.form['months_in_repayment'])
        channel = request.form['channel']
        occupancy = request.form['occupancy']
        property_type = request.form['property_type']
        original_interest_rate = float(request.form['original_interest_rate'])
        original_loan_term = int(request.form['original_loan_term'])  
        units = int(request.form['units'])
        number_of_borrowers = int(request.form['number_of_borrowers'])
        mip = int(request.form['mip'])
        original_upb = float(request.form['original_upb'])
        ocltv = float(request.form['ocltv'])

        result = predictor.mortgage_pipeline(
            credit_score, dti, months_in_repayment, channel, occupancy,
            property_type, original_interest_rate, original_loan_term,
            units, number_of_borrowers, mip, original_upb, ocltv
        )
        return render_template('result.html', result=result)
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    #application .run(debug=True)
    application .run(host="0.0.0.0",port=5000)