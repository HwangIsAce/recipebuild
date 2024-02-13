import bootstrap
from datasets import load_dataset
from tokenizers import (Tokenizer, decoders, models, normalizers,
                        pre_tokenizers, processors, trainers)
from transformers import PreTrainedTokenizerFast

class RBTokenizer:
    def __init__(
            self, 
            rb_config
        ):
        self.rb_tokenizer = rb_config.bert_config['tokenizer']
        self.rb_config = rb_config.bert_config
        
    def __str__(self) -> str:
        return f"Custom Tokenizer = {self.rb_tokenizer}"
    
    def __repr__(self) -> str:
        return self.__str__()

    def _get_training_corpus(self, ds, split='train', batch_size=100):
        _current_ds = ds[split]
        for i in range(0, len(_current_ds), batch_size):
            yield _current_ds[i : i + batch_size]["text"]
        
    def train(self):
        # _tokenizer : internal use only
        self._tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

        self._tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
 
        self._tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        
        # data path to train the tokenizer
        _train_file_path = self.rb_tokenizer["tokenizer_train_file_path"]["ingr_title_tag"]

        # load dataset to train the tokenizer
        self.rb_ds = load_dataset("text", data_files={"train":_train_file_path})
        
        special_tokens = self.rb_config['tokenizer']["special_tokens"]
        
        trainer = trainers.WordPieceTrainer(vocab_size=self.rb_config['tokenizer']["vocab_max_size"], special_tokens=special_tokens)
        
        self._tokenizer.train_from_iterator(self._get_training_corpus(self.rb_ds), trainer=trainer)

    def save(self):
        if self._tokenizer is None:
            raise ValueError('Tokenizer is None, maybe train first?')
        self._tokenizer.save(self.rb_tokenizer["tokenizer_file_path"]['ingr_title_tag'])
        
    def load(self):
        self.tokenizer = PreTrainedTokenizerFast(
            # tokenizer_object=self._tokenizer,
            tokenizer_file=self.rb_tokenizer["tokenizer_file_path"]['ingr_title_tag'], # You can load from the tokenizer file, alternatively
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            max_len=self.rb_tokenizer["max_len"],
        )


# if __name__ == "__main__":
#     ingt_config = bootstrap.IngTConfig(vocab='ingr_only',path="/media/ssd/dh/projects/ing_mlm/config.json")
#     ingt_tokenizer = IngtTokenizer(ingt_config) 
#     ingt_tokenizer.train()
#     ingt_tokenizer.save()
#     # ingt_test_ds['train'][0]
#     ingt_tokenizer.load()
    
    