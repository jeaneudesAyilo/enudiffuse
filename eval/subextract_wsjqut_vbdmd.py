import json
import numpy as np 


wsj_full_data = "./eval/wsj_test.json"

with open(wsj_full_data, "r") as f:
    dataset = json.load(f)

listings = [dataset[key] for key in dataset.keys()]

np.random.seed(42)

listings_minus5 = [dataset[key] for key in dataset.keys() if dataset[key]["snr"]==-5]

listings_0 = [dataset[key] for key in dataset.keys() if dataset[key]["snr"]==0]

listings_plus5 = [dataset[key] for key in dataset.keys() if dataset[key]["snr"]==5]


sample_listings_minus5 = list(np.random.choice(listings_minus5, size=20, replace=False,))

sample_listings_0 = list(np.random.choice(listings_0, size=20, replace=False,))

sample_listings_plus5 = list(np.random.choice( listings_plus5, size=20, replace=False,))


sample_listings = []

sample_listings.extend(sample_listings_minus5)
sample_listings.extend(sample_listings_0)
sample_listings.extend(sample_listings_plus5)

sample_dict = {element["utt_name"]:element for element in sample_listings}


sample_dict_path = "./eval/sample_wsj0qut_rtf.json"

with open(sample_dict_path, "w") as file:
    json.dump(sample_dict, file,indent=2)



###================= VBDMD


vb_full_data = "./eval/vb_dmd.json"

with open(vb_full_data, "r") as f:
    dataset = json.load(f)


listings = [dataset[key] for key in dataset.keys()]

np.random.seed(42)

listings_2_5 = [dataset[key] for key in dataset.keys() if (dataset[key]["snr"]==2.5) & (dataset[key]["length"]>2)]

listings_7_5 = [dataset[key] for key in dataset.keys() if (dataset[key]["snr"]==7.5) & (dataset[key]["length"]>2)]

listings_12_5 = [dataset[key] for key in dataset.keys() if (dataset[key]["snr"]==12.5) & (dataset[key]["length"]>2)]

listings_17_5 = [dataset[key] for key in dataset.keys() if (dataset[key]["snr"]==17.5) & (dataset[key]["length"]>2)]


sample_listings_2_5 = list(np.random.choice(listings_2_5, size=15, replace=False,))

sample_listings_7_5 = list(np.random.choice(listings_7_5, size=15, replace=False,))

sample_listings_12_5 = list(np.random.choice(listings_12_5, size=15, replace=False,))

sample_listings_17_5 = list(np.random.choice(listings_17_5, size=15, replace=False,))


sample_listings_vb = []

sample_listings_vb.extend(sample_listings_2_5)
sample_listings_vb.extend(sample_listings_7_5)
sample_listings_vb.extend(sample_listings_12_5)
sample_listings_vb.extend(sample_listings_17_5)


sample_dict_vb = {element["utt_name"]:element for element in sample_listings_vb}

sample_dict_path_vb = "./eval/sample_vbdmd_rtf.json"

with open(sample_dict_path_vb, "w") as file:
    json.dump(sample_dict_vb, file,indent=2)


print("####### succesfully ended #######")