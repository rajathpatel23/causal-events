import torch
from src.model import ContrastiveModel
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import json

if __name__ == '__main__':
    base_model="roberta-base"
    model_name = "causal-news-256-128-5e-5-0.07-5-False-2023-08-23-opps-roberta-base"
    model_directory="/home/jovyan/work/causal-events/src/report/contrastive"
    model_path=f"{model_directory}/{model_name}"
    max_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pooling = True
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = ContrastiveModel(len_tokenizer=len(tokenizer)+2, model=base_model).to(device)
    model.load_state_dict(
        torch.load(
            "{}/pytorch_model.bin".format(model_path),
            map_location=torch.device(device),
        ),
        strict=False,
    )
    test_data = pd.read_csv("/home/jovyan/work/causal-events/data/subtask1/dev_subtask1.csv")
    test_data = test_data.to_dict(orient="records")
    test_data_array = []
    label_list = []

    for index, data in tqdm(enumerate(test_data)):
        entity_str = data['text']
        label_list.append(data['label'])
        entity_tokenizer = tokenizer([entity_str],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length).to(device)
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            _, outputs = model(input_ids=entity_tokenizer["input_ids"], attention_mask=entity_tokenizer["attention_mask"])
            outputs = outputs.squeeze().tolist()
        test_data_array.append(outputs)

test_data_array = np.array(test_data_array)
X_embedded = TSNE(n_components=2, early_exaggeration=2, learning_rate='auto', init='pca', perplexity=30).fit_transform(test_data_array)
print(X_embedded.shape)
# import pdb; pdb.set_trace()
# kmeans =KMeans(n_clusters=2)
# labels_list = kmeans.fit_predict(X_embedded)
df = pd.DataFrame()
df['y'] = label_list
df["comp-1"] = X_embedded[:,0]
df["comp-2"] = X_embedded[:,1]

# output_predict_file = os.path.join(f"predict_results_kmeans_causal_news.json")
# with open(output_predict_file, "w") as writer:
#     for index, item in enumerate(labels_list):
#         dict_data = {"index": index, "prediction": item.item()}
#         writer.write(f"{json.dumps(dict_data)}\n")


ax = sns.scatterplot(x="comp-1", y="comp-2",
                hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="Causal News Test Corpus T-SNE projection")
# save the plot as PNG file
plt.savefig("seaborn_plot_dev.png")


# _, inputs = tokenizer(
#     entity_strs,
#     return_tensors="pt",
#     padding=True,
#     truncation=True,
#     max_length=self.max_length,
# ).to(self.device)

