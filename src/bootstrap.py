import json
import os
import pickle
import time

os.environ["TZ"] = "Asia/Seoul"
time.tzset()

class recipebuildConfig:

    def __init__(
        self,
        path = "/home/dongee/projects/mlm_tmp/mlm/config.json" # config file path
    ):
        self.path = path
        
        with open(self.path, 'r') as f:
            self.static_config = json.load(f) # config file

        # data path
        self.data_config = self.static_config['data_folder']

        # before data processing (original data path)
        self.original_data_folder = f"{self.static_config['data_folder']['home_dir']}/{self.static_config['data_folder']['original_dir']}"

        # after data precessing (processed data path)
        self.processed_data_folder = f"{self.static_config['data_folder']['home_dir']}/{self.static_config['data_folder']['processed_dir']}"

        # tokenizer 
        self.tokenizer = self.static_config['bert_config']['tokenizer']['tokenizer_file_path']['ingr_title_tag']

        # bert config
        self.bert_config = self.static_config['bert_config']
        
        # bert embedding resize
        self.bert_resize_embedding = self.static_config['bert_config']['resize_embedding']

        def get_dictionary(self):
            with open(f"{self.original_data_folder}//iid2ingr_full.pkl", "rb") as f:
                self.iid2ingr_data = pickle.load(f)
            self.ing_list = list(self.iid2ingre_data.values())
        
        def __str__(self):
            return f"path = {self.path} / static_config = {self.static_config} / original_data_folder = {self.original_data_folder}"

        def __repr__(self):
            return self.__str__()


if __name__ == "__main__":
    print(" >> Config")
    config = recipebuildConfig()
    print(config)
