import h2o

h2o.init()

# Load models
model1_path = r"C:\Users\Jaival Singh\Downloads\Mlops Project\downloaded_models\1_models_m-0734872714564704839db8800d90d9cb_artifacts_model.h2o_DeepLearning_model_python_1765071240905_3"
model1 = h2o.load_model(model1_path)

model2_path = r"C:\Users\Jaival Singh\Downloads\Mlops Project\downloaded_models\1_models_m-3e2c075b7fe64b04b6c670813a90d45f_artifacts_model.h2o_StackedEnsemble_model_python_1765071240905_11"
model2 = h2o.load_model(model2_path)

model3_path = r"C:\Users\Jaival Singh\Downloads\Mlops Project\downloaded_models\1_models_m-4b63a40654854820829f7a44cd14313b_artifacts_model.h2o_StackedEnsemble_model_python_1765071240905_15"
model3 = h2o.load_model(model3_path)

# Function to safely get feature names
def get_features(model):
    try:
        return model.var_names
    except AttributeError:
        # fallback for some H2O models
        return model._model_json['output']['names']

# Print features for each model
print("Model1 features:", get_features(model1))
print("Model2 features:", get_features(model2))
print("Model3 features:", get_features(model3))
