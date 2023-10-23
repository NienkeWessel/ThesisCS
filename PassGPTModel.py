from MLModel import MLModel
from transformers import RobertaTokenizerFast
import torch

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
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


class GPTModel(MLModel):
    def __init__(self) -> None:
        pass

    def train(self, X, y):
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


class PassGPT10Model(GPTModel):
    def __init__(self, internet=True) -> None:
        self.internet = internet
        if internet:
            model_loc = "javirandor/passgpt-10characters"
        else:
            model_loc = "passgpt-10characters"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_loc, num_labels=2)

        seed = 42
        torch.manual_seed(seed)

        # Define training arguments
        self.training_args = TrainingArguments(
            output_dir="./model",
            evaluation_strategy="steps",
            eval_steps=100,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            save_steps=500,
            save_total_limit=2,
            seed=seed,
            report_to='none'
        )

        self.tokenizer = tokenizer = RobertaTokenizerFast.from_pretrained(model_loc,
                                                                          max_len=12, padding="max_length",
                                                                          truncation=True, do_lower_case=False,
                                                                          strip_accents=False, mask_token="<mask>",
                                                                          unk_token="<unk>", pad_token="<pad>",
                                                                          truncation_side="right",
                                                                          is_split_into_words=True)

    def train(self, X, y):

        X.update({'labels': y})
        dataset = HuggingfaceDataset.from_dict(X)

        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.trainer.train()

    def predict(self, X):

        dataset = HuggingfaceDataset.from_dict(X)
        return self.trainer.predict(dataset)
