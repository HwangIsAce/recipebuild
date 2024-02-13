import os
import pickle
from pathlib import Path
import json

import bootstrap
from loguru import logger
from tqdm import tqdm

###  process the original data and store it in processed  ###

rb_config = bootstrap.recipebuildConfig(
    path = "/home/dongee/projects/mlm_tmp/mlm/config.json"
)

# data path 
data_config = rb_config.data_config

# origin data path
original_data_path = rb_config.original_data_folder

# log
logger.info(f"original_file_path = {original_data_path}")
logger.info(f"data_config = {data_config}")

with open(f"{original_data_path}/iids_full.pkl", "rb") as f:
    iids_data = pickle.load(f)

with open(f"{original_data_path}/iid2ingr_full.pkl", "rb") as f:
    iid2ingr_data = pickle.load(f)

with open(f"{original_data_path}/iid2fgvec_full.pkl", "rb") as f:
    iids2fgvec_data = pickle.load(f)

with open(f"{original_data_path}/rid2title.pkl", "rb") as f:
    rid2title_data = pickle.load(f)

# split info
with open(f"{original_data_path}/rid2info_full.pkl", "rb") as f:
    rid2info_full = pickle.load(f)

with open(f"{original_data_path}/rid2sorted_iids_full.pkl", "rb") as f:
    rid2sorted_iids_data = pickle.load(f)

with open(f"{original_data_path}/tags.json", "rb") as f:
    rid2tags = json.load(f)

# processed data path
processed_data_path = f"{data_config['home_dir']}/{data_config['processed_dir']}"

logger.info(f"target_dir : {processed_data_path}")

class ProcessHandler:
    def __init__(
        self,
        output_path,
        rid2info_full,
        iid2ingr_data,
        rid2sorted_iids_data,
        ver="version_ingr_title_tag",
    ):
        self.output_path = output_path
        Path(output_path).mkdir(parents=False, exist_ok=True)
        self.rid2info_full = rid2info_full
        self.ver = ver

        # 
        if self.ver.startswith("version_ingr_title_tag"):
            mode = "ingr_title_tag"
        else:
            raise ValueError(f"Invalid ver = {ver}")

        if self.ver.startswith("version_ingr_title_tag"):
            self.train_handler = PartitionProcessHandler(
                output_path,
                iid2ingr_data,
                rid2sorted_iids_data,
                "train",
                mode=mode,
            )
            self.val_handler = PartitionProcessHandler(
                output_path,
                iid2ingr_data,
                rid2sorted_iids_data,
                "val",
                mode=mode,
            )
            self.test_handler = PartitionProcessHandler(
                output_path,
                iid2ingr_data,
                rid2sorted_iids_data,
                "test",
                mode=mode,
            )
        else:
            raise ValueError(f"Invalid ver = {ver}")

        if ver.endswith("_sample"):
            self.sample = True
        else:
            self.sample = False
        # ingr_title -> not now

    def write(self, rid):
        # if rid -> partion -> handler write
        # if handler.cnt > 00 - sample
        _partition = self.rid2info_full[rid]["partition"]
        if _partition == "train":
            _handler = self.train_handler
        elif _partition == "val":
            _handler = self.val_handler
        elif _partition == "test":
            _handler = self.test_handler
        else:
            raise ValueError(f"Invalid partition = {_partition}")

        if self.sample and _handler.write_cnt > 300:
            return False
        _handler.write(rid)
        return True

    def close(self):
        self.train_handler.close()
        self.val_handler.close()
        self.test_handler.close()


class PartitionProcessHandler:
    def __init__(
        self,
        output_path,
        iid2ingr_data,
        rid2sorted_iids_data,
        partition="train",
        mode="ingr_title_tag",
    ):
        self.partition = partition
        self.mode = mode

        self.output_path = output_path
        self.iid2ingr_data = iid2ingr_data
        self.rid2sorted_iids_data = rid2sorted_iids_data

        self.path = os.path.join(self.output_path, f"{partition}.txt")
        self.f = open(self.path, "w")
        self.write_cnt = 0

    def write(self, rid):

        ingr_txt = " ".join(
            self.iid2ingr_data[int(_iid)] for _iid in self.rid2sorted_iids_data[rid]
        )
        tag_txt = " ".join(
            _iid for _iid in self.rid2tags[rid] 
        ) #js

        if self.mode == "ingr_title_tag":
            sen = f"{ingr_txt}[SEP]{tag_txt}[SEP]{title}"
        else:
            raise ValueError(f"Invalid Format model = {self.mode}")

        self.f.write(f"{sen}\n")
        self.write_cnt += 1

    def close(self):
        self.f.close()


# 
ing_title_tag_path = f"{processed_data_path}/v3_ing_title_tag" # Check path before creating data
logger.info(f"v3_ing_title_tag_path = {ing_title_tag_path }")
ing_title_tag_handler = ProcessHandler(
    ing_title_tag_path,
    rid2info_full,
    iid2ingr_data,
    rid2sorted_iids_data,
    ver="version_ingr_title_tag",
)
ing_title_tag_sample_path = f"{processed_data_path}/v3_ing_title_tag_sample" # Check path before creating data
logger.info(f"v3_ing_title_tag_sample_path = {ing_title_tag_sample_path }")
ing_title_tag_sample_handler = ProcessHandler(
    ing_title_tag_sample_path,
    rid2info_full,
    iid2ingr_data,
    rid2sorted_iids_data,
    ver="version_ingr_title_tag_sample",
)

error_rid_f = open(f"{processed_data_path}/error_rids.txt", "w")

for idx, (rid, title) in tqdm(enumerate(rid2title_data.items())):

    if rid not in rid2sorted_iids_data:
        error_rid_f.write(f"{rid}\n")
        continue

    ing_title_tag_handler.write(rid)
    ing_title_tag_sample_handler.write(rid)

ing_title_tag_handler.close()
ing_title_tag_sample_handler.close()

error_rid_f.close()
