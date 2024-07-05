import pickle
import numpy as np
#-------------------------------------------------------------------------------------
class prepayment_risk_prediction:
    def __init__(self):
                    # Loading encoders, scalers, and models
                    with open('property_type_encoder.pkl', 'rb') as file:
                        self.property_type_encoder = pickle.load(file)
                    
                    with open('occupancy_encoder.pkl', 'rb') as file:
                        self.occupancy_encoder = pickle.load(file)
                    
                    with open('channel_encoder.pkl', 'rb') as file:
                        self.channel_encoder = pickle.load(file)
                    
                    with open('classification_standard_scaler.pkl', 'rb') as file:
                        self.classification_standard_scaler = pickle.load(file)
                    
                    with open('regression_standard_scaler.pkl', 'rb') as file:
                        self.regression_standard_scaler = pickle.load(file)
                    
                    with open('logistic_regression_model.pkl', 'rb') as file:
                        self.logistic_regression_model = pickle.load(file)
                    
                    with open('linear_regression_model.pkl', 'rb') as file:
                        self.linear_regression_model = pickle.load(file)
    #------------------------------------------------------------------------------
    #Function to convert credit score to credit score range
    def credit_score_range_encoded(self, credit_score):
                    if 0>=credit_score<=649:
                        return 0
                    elif 650>=credit_score<=699:
                        return 1
                    elif 700>=credit_score<=749:
                        return 2
                    elif 750>=credit_score<=900:
                        return 3  
                    else:
                        return 3    
    #------------------------------------------------------------------------------
    #Function to convert DTI score to DTI range
    def dti_range_encoded(self, dti):
                    if dti > 40:
                        return 2
                    elif 20 <= dti <= 40:
                        return 1
                    else:
                        return 0
    #------------------------------------------------------------------------------
    #Function to convert years in repayment to years in repayment range
    def years_in_repayment_range_encoded(self, months_in_repayment):
                    #Converting months to years
                    years_in_repayment= months_in_repayment/ 12
                    #----------------------------------------------
                    if years_in_repayment >= 16:
                        return 4
                    elif 12 <= years_in_repayment < 16:
                        return 3
                    elif 8 <= years_in_repayment < 12:
                        return 2
                    elif 4 <= years_in_repayment < 8:
                        return 1
                    elif 0>= years_in_repayment <= 4:
                        return 0
                    else:
                        return 4
    #------------------------------------------------------------------------------
    #Function to convert property type to encoded format
    def property_type_encoded(self, property_type):
                    property_type=property_type.lower()
                    property_type_map={'single family': "SF", "planned unit development": "PU", "condominium": "CO", "Manufactured Home": "MH", "leasehold": "LH", "cooperative": "CP" }
                    property_type_encoded=self.property_type_encoder.transform([property_type_map[property_type]])
                    return property_type_encoded[0]
    #------------------------------------------------------------------------------
    #Function to convert occupancy to encoded format
    def occupancy_encoded(self, occupancy):
                    occupancy=occupancy.lower()
                    occupancy_map={"owner occupied": "O", 'investment': "I", 'second home': "S"}
                    occupancy_encoded=self.occupancy_encoder.transform([occupancy_map[occupancy]])
                    return occupancy_encoded[0]
    #------------------------------------------------------------------------------
    #Function to convert occupancy to encoded format
    def channel_encoded(self, channel):
                    channel=channel.lower()
                    channel_map={"third party organisation": "T", 'retail': "R", "correspondent":"C", 'broker': "B"}
                    channel_encoded=self.channel_encoder.transform([channel_map[channel]])
                    return channel_encoded[0]
    #------------------------------------------------------------------------------
    #Function to perform regression
    def regression(self, regression_input_features):
                    #Normalizing the data
                    regression_input_features_scaled = self.regression_standard_scaler.transform(regression_input_features)
                    #Prediction
                    predicted_output = self.linear_regression_model.predict(regression_input_features_scaled)
                    #Converting the output into relevant format
                    predicted_output = np.exp(predicted_output)
                    # return round(predicted_output[0], 2)
                    return f"The customer has no history of loan delinquency.The customer has a prepayment risk of {round(predicted_output[0], 2)}%"
    #------------------------------------------------------------------------------
    #Function to perform classifiaction
    def classification(self, classification_input_features, regression_input_features):
                    #Normalizing the data
                    classification_input_features_scaled = self.classification_standard_scaler.transform(classification_input_features)
                    #Prediction
                    predicted_output = self.logistic_regression_model.predict(classification_input_features_scaled)
                    #---------------------------------------
                    if predicted_output == 1:
                        return "The customer loan is Delinquent"
                    else:
                        return self.regression(regression_input_features)
    #------------------------------------------------------------------------------
    def mortgage_pipeline(self, credit_score, dti, months_in_repayment, channel, occupancy, property_type, original_interest_rate, original_loan_term, units, number_of_borrowers, mip, original_upb, ocltv):
                    # Preprocess inputs
                    credit_score_encoded = self.credit_score_range_encoded(credit_score)
                    dti_encoded = self.dti_range_encoded(dti)
                    years_in_repayment_encoded = self.years_in_repayment_range_encoded(months_in_repayment)
                    property_type_encoded_val = self.property_type_encoded(property_type)
                    occupancy_encoded_val = self.occupancy_encoded(occupancy)
                    channel_encoded_val = self.channel_encoded(channel)
                    #Classification input features
                    classification_input_features = [[credit_score_encoded, years_in_repayment_encoded, property_type_encoded_val, occupancy_encoded_val, original_loan_term, dti_encoded, units, number_of_borrowers, mip, channel_encoded_val, original_interest_rate]]
                    
                    #Regression input features
                    regression_input_features = [[credit_score_encoded, years_in_repayment_encoded, original_upb, occupancy_encoded_val, original_loan_term, dti_encoded, original_interest_rate, number_of_borrowers, mip, ocltv, channel_encoded_val]]
                    #Perform classification or regression based on prediction
                    return self.classification(classification_input_features, regression_input_features)