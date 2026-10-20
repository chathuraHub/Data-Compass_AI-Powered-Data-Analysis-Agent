import os, numpy as np, pandas as pd
from src.Airbnb.utils.utils import load_object

D = "airbnb-price-prediction"
test = pd.read_csv(os.path.join(D, "Airbnb_test.csv"))

# same cleaning as training
test['cleaning_fee'] = test['cleaning_fee'].astype(str).str.upper()
test['host_response_rate'] = pd.to_numeric(
    test['host_response_rate'].astype(str).str.replace('%', '', regex=False), errors='coerce')
test['amenities'] = [len(str(a).split(',')) for a in test['amenities']]

pre = load_object(os.path.join(D, "Preprocessor.pkl"))
model = load_object(os.path.join(D, "Model.pkl"))

preds = np.exp(model.predict(pre.transform(test)))

sub = pd.DataFrame({"id": test["id"], "price": preds})   # match sample_submission columns
sub.to_csv(os.path.join(D, "my_submission.csv"), index=False)
print(sub.head())