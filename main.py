import os

os.system("python src/data_ingestion.py")
print('Data Ingestion done')
os.system("python src/data_preprocessing.py")
print('data preprocessing done')
os.system("python src/data_feature_scaling.py")
print("scaling done")
os.system("python src/data_modeling.py")
print("data modelling done")
os.system("python src/data_evaluation.py")
print('data evaluation done')