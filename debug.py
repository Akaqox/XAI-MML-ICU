from keras.models import Model, model_from_json

" load lung segment model and weights... (model U-net) " 
json_file = open('../vendor/segment_model.json', 'r') 
loaded_model_json = json_file.read() 
json_file.close() 
model = model_from_json(loaded_model_json)
    
" load weights into the model " 
model.load_weights('../vendor/segment_model.h5') 
print("Loaded model from disk")


model.compile()

# Save the model in `.keras` format
model.save('segment_model.keras', save_format='keras')