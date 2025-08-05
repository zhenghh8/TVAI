import json
import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import io

class COCODataSet(Dataset):
    def __init__(self, data_path, trans, model):
        self.data_path = data_path
        self.trans = trans
        self.model = model

        img_files = os.listdir(self.data_path)
        random.shuffle(img_files)
        self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img_id = int(img_file.split(".jpg")[0][-6:])
        img_path = os.path.join(self.data_path, img_file)
        
        if self.model == 'qwen-vl-chat':
            image = img_path
        else:
            image = Image.open(img_path).convert("RGB")
            image = self.trans(image)

        return {"img_id": img_id, "image": image}


class POPEDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans, model):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans
        self.model = model

        image_list, query_list, label_list = [], [], []
        for q in open(pope_path, "r"):
            line = json.loads(q)
            image_list.append(line["image"])
            query_list.append(line["text"])
            label_list.append(line["label"])

        for i in range(len(label_list)):
            if label_list[i] == "no":
                label_list[i] = 0
            else:
                label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        if self.model == 'qwen-vl-chat':
            raw_image = image_path
        else:
            raw_image = Image.open(image_path).convert("RGB")
        image = self.trans(raw_image)
        query = self.query_list[index]
        label = self.label_list[index]

        return {"image": image, "query": query, "label": label}


class POPEChatDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans, model):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans
        self.model = model

        image_list, query_list, label_list = [], [], []

        for q in open(pope_path, "r"):
            line = json.loads(q)
            image_list.append(line["image"])
            query_list.append(line["text"])
            label_list.append(line["label"])

        for i in range(len(label_list)):
            for j in range(len(label_list[i])):
                if label_list[i][j] == "no":
                    label_list[i][j] = 0
                else:
                    label_list[i][j] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        if self.model == 'qwen-vl-chat':
            raw_image = image_path
        else:
            raw_image = Image.open(image_path).convert("RGB")
        image = self.trans(raw_image)
        query = self.query_list[index]
        label = self.label_list[index]
        image_id = self.image_list[index]

        return {"image": image, "query": query, "label": label, "image_id": image_id}

class MMBenchDataSet(Dataset):
    def __init__(self, data_path, trans, model):
        self.data_path = data_path
        self.trans = trans
        self.model = model

        index_list, image_list, query_list, label_list = [], [], [], []

        for q in open(data_path, "r"):
            line = json.loads(q)
            index_list.append(line["index"])
            image_list.append(line["image"])
            query_list.append(line["text"])
            label_list.append(line["label"])

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.index_list = index_list
        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_list[index])
        if self.model == 'qwen-vl-chat':
            raw_image = image_path
        else:
            raw_image = Image.open(image_path).convert("RGB")
        index_ = self.index_list[index]
        image = self.trans(raw_image)
        query = self.query_list[index]
        label = self.label_list[index]

        return {"index": index_, "image": image, "query": query, "label": label}
