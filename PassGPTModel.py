import pandas as pd

from MLModel import MLModel
from utils import confusion

from transformers import RobertaTokenizerFast
import torch
from abc import ABC, abstractmethod


from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import ReformerConfig, ReformerForSequenceClassification
from datasets import Dataset as HuggingfaceDataset


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
        return torch.mean(correct).tolist()


    def transform_pred(self, pred, y):
        y_hat = torch.transpose(
            torch.vstack(((pred[:, 0] > pred[:, 1]).unsqueeze(0), (pred[:, 0] <= pred[:, 1]).unsqueeze(0))), 0, 1)
        return (y_hat >= 0.5).to(y.dtype)


    def calc_recall(self, y, pred):
        pred = torch.tensor(pred.predictions)
        pred = self.transform_pred(pred, y)
        tp, fp, tn, fn = confusion(pred, y)
        return tp / (tp + fn)

    def calc_precision(self, y, pred):
        pred = torch.tensor(pred.predictions)
        pred = self.transform_pred(pred, y)
        tp, fp, tn, fn = confusion(pred, y)
        return tp / (tp+fp)

    def calc_f1score(self, y, pred):
        pred = torch.tensor(pred.predictions)
        pred = self.transform_pred(pred, y)
        tp, fp, tn, fn = confusion(pred, y)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return (2 * precision * recall) / (precision + recall)

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
    def __init__(self, params) -> None:
        super().__init__(params)
        self.internet = params['model_params']['internet']
        if self.model_loc is not None:
            model_loc = self.model_loc
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
        if 'loss_file' in params:
            loss_file = params['loss_file']
        else:
            loss_file = "lossPassGPT.csv"
        loss_history = pd.DataFrame(self.trainer.state.log_history)
        loss_history.to_csv(loss_file)

    def predict(self, X):
        dataset = HuggingfaceDataset.from_dict(X)
        return self.trainer.predict(dataset)

    def save_model(self, filename):
        super().save_model(filename)
        self.tokenizer.save_pretrained(filename)

    def load_model(self, filename):
        ''' WARNING! Does not seem to work properly; loaded model does not override the model created
        at initialization of the object completely

        :param filename:
        :return:
        '''
        return
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
    def __init__(self, params) -> None:
        super().__init__(params)
        self.internet = params['model_params']['internet']
        if self.model_loc is not None:
            model_loc = self.model_loc
        elif self.internet:
            model_loc = "google/reformer-enwik8"
        else:
            model_loc = "reformer-enwik8"
        conf = ReformerConfig.from_pretrained(model_loc)
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
        dataset = HuggingfaceDataset.from_dict(X)
        return self.trainer.predict(dataset)

    def load_model(self, filename):
        ''' WARNING! Does not seem to work properly; loaded model does not override the model created
        at initialization of the object completely

        :param filename:
        :return:
        '''
        return
        conf = ReformerConfig.from_pretrained(filename)
        self.model = ReformerForSequenceClassification.from_pretrained(filename, config=conf)

    def __str__(self) -> str:
        return "ReformerModel"