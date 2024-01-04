from MLModel import MLModel
from transformers import RobertaTokenizerFast
import torch
from abc import ABC, abstractmethod

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import ReformerConfig, ReformerForSequenceClassification
from datasets import Dataset as HuggingfaceDataset
import evaluate



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'],
                                            inputs['labels'])
        return (loss, outputs) if return_outputs else loss


class HuggingfaceModel(MLModel):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.trainer = None

    def train(self, X, y, params=None):
        pass

    def predict(self, X):
        pass

    def calc_accuracy(self, y, pred):
        pred = torch.tensor(pred.predictions)
        y_hat = torch.transpose(
            torch.vstack(((pred[:, 0] > pred[:, 1]).unsqueeze(0), (pred[:, 0] <= pred[:, 1]).unsqueeze(0))), 0, 1)
        y_hat = (y_hat >= 0.5).to(y.dtype)
        correct = (y_hat == y).to(torch.float32)
        return torch.mean(correct)

    def calc_recall(self, y, pred):
        pass

    def calc_precision(self, y, pred):
        pass

    def calc_f1score(self, y, pred):
        pass
        '''f1_metric = evaluate.load("f1")
        results = f1_metric.compute(predictions=pred, references=y)
        return results
        '''

    def save_model(self, filename):
        #self.trainer.save_model(filename)
        self.model.save_pretrained(filename)

    @abstractmethod
    def load_model(self, filename):
        """
        This one is not defined as the individual models should define it
        :param filename:
        :return:
        """
        pass


class PassGPT10Model(HuggingfaceModel):
    def __init__(self, params, load_filename=None) -> None:
        super().__init__(params)
        self.internet = params['model_params']['internet']
        if load_filename is not None:
            model_loc = load_filename
        elif self.internet:
            model_loc = "javirandor/passgpt-10characters"
        else:
            model_loc = "passgpt-10characters"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_loc, num_labels=2)
        self.model.config.pad_token_id = self.model.config.eos_token_id

        seed = 42
        torch.manual_seed(seed)

        # Define training arguments
        self.training_args = TrainingArguments(
            output_dir="./model",
            evaluation_strategy="steps",
            eval_steps=0.1, # run evaluation at every 10% of the dataset
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            save_steps=500,
            save_total_limit=2,
            seed=seed,
            report_to='none'
        )

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_loc,
                                                              max_len=12, padding="max_length",
                                                              truncation=True, do_lower_case=False,
                                                              strip_accents=False, mask_token="<mask>",
                                                              unk_token="<unk>", pad_token="<pad>",
                                                              truncation_side="right",
                                                              is_split_into_words=True)

        self.trainer = CustomTrainer(model=self.model, args=self.training_args, tokenizer=self.tokenizer,)
    
    def __str__(self) -> str:
        return "PassGPTModel"

    def train(self, X, y, params=None):
        X.update({'labels': y})
        dataset = HuggingfaceDataset.from_dict(X)

        dataset = dataset.train_test_split(test_size=0.1)

        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            tokenizer=self.tokenizer,
        )

        self.trainer.train()

    def predict(self, X):
        print(X)
        dataset = HuggingfaceDataset.from_dict(X)
        return self.trainer.predict(dataset)

    def save_model(self, filename):
        super().save_model(filename)
        self.tokenizer.save_pretrained(filename)

    def load_model(self, filename):
        self.model = AutoModelForSequenceClassification.from_pretrained(filename, num_labels=2)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(filename,
                                                              max_len=12, padding="max_length",
                                                              truncation=True, do_lower_case=False,
                                                              strip_accents=False, mask_token="<mask>",
                                                              unk_token="<unk>", pad_token="<pad>",
                                                              truncation_side="right",
                                                              is_split_into_words=True)




# https://huggingface.co/viklofg/swedish-ocr-correction
# https://huggingface.co/docs/transformers/model_doc/reformer -> should be more efficient for longer entry sequences, which is what we want
        # https://huggingface.co/google/reformer-enwik8
        # https://huggingface.co/robingeibel/reformer-finetuned/blob/main/README.md (nul informatie beschikbaar)
# Tay et al. compared different transformer variants that should be better for longer input sequences
        
class ReformerModel(HuggingfaceModel):
    # NB: https://stackoverflow.com/questions/68742863/error-while-trying-to-fine-tune-the-reformermodelwithlmhead-google-reformer-enw
    def __init__(self, params, load_filename=None) -> None:
        super().__init__(params)
        self.internet = params['model_params']['internet']
        if load_filename is not None:
            model_loc = load_filename
        elif self.internet:
            model_loc = "google/reformer-enwik8"
        else:
            model_loc = "reformer-enwik8"
        conf = ReformerConfig.from_pretrained('google/reformer-enwik8')
        conf.axial_pos_embds = False 
        self.model = AutoModelForSequenceClassification.from_pretrained(model_loc, config =conf)

        seed = 42
        torch.manual_seed(seed)

        # Define training arguments
        self.training_args = TrainingArguments(
            output_dir="./reformermodel",
            evaluation_strategy="steps",
            eval_steps=0.1, # run evaluation at every 10% of the dataset
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            save_steps=500,
            save_total_limit=2,
            seed=seed,
            report_to='none'
        )

        self.trainer = CustomTrainer(model=self.model, args=self.training_args,)
    

    def train(self, X, y, params=None):
        X.update({'labels': y})

        dataset = HuggingfaceDataset.from_dict(X)

        dataset = dataset.train_test_split(test_size=0.1)

        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
        )

        self.trainer.train()

    def predict(self, X):
        print(X)
        dataset = HuggingfaceDataset.from_dict(X)
        return self.trainer.predict(dataset)

    def load_model(self, filename):
        self.model = ReformerForSequenceClassification.from_pretrained(filename, num_labels=2)

    def __str__(self) -> str:
        return "ReformerModel"