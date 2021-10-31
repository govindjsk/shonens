import copy
from datetime import datetime
import json
import os

from utils import exceptions

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class BaseModel(torch.nn.Module):
    """Base model containing common methods which are useful across
    all models."""
    def __init__(self):
        super(BaseModel, self).__init__()
        self.writer = SummaryWriter('./.logs')
        self.instance_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.variable_name_prefix = '%s/%s' % (
            self.__class__.__name__, self.instance_time)
        self.reports = []
        self.prev_best_score = 0
        self.best_state_dict = None
    
    def get_report(self, y_true, y_pred):
        """Generates model report from true and predicted labels.

        Args:
            y_true: torch.Tensor. True outputs.
            y_pred;: torch.Tensor. Predicted outputs.
        
        Returns:
            dict. A dictionary comprising of several evaluation metrics.
        """
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        roc_score = roc_auc_score(y_true, y_pred)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        report = {
            'ROC': roc_score,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred)
        }
        return report
    
    def add_report(self, train_report, test_report=None, track_model=True):
        """Adds report to report list so that it can be saved later for further
        analysis.

        Args:
            train_report: dict.
            test_report: dict|None.
        """
        if test_report:
            self.reports.append({
                'train_report': train_report,
                'test_report': test_report
            })
        else:
            self.reports.append({
                'train_report': train_report
            })
    
    def track_best_model(self, score):
        """Keeps track of best model yet based on score value."""
        if self.prev_best_score < score:
            self.prev_best_score = score
            self.best_state_dict = copy.deepcopy(self.state_dict())
    
    def save_best_model(self, dataset):
        """Stores best model as pt file."""
        if not os.path.exists('./models/%s/' % dataset):
            os.makedirs('./models/%s/' % dataset)
        torch.save(
            self.best_state_dict,
            './models/%s/%s-%s.pt' % (
                dataset, self.__class__.__name__, self.instance_time))
    
    def load_best_model(self):
        """Loads the best model yet."""
        if self.best_state_dict:
            print('Loading best model')
            self.load_state_dict(self.best_state_dict)
        else:
            raise exceptions.BestModelNotTrackedError

    def print_report(self, report, epoch_count, epoch_loss, train=True):
        """Prints the model report to the screen.

        Args:
            report: dict. A dictionary that contains model report comprising
                of several evaluation metrics.
            epoch_count: int. Current epoch number.
            epoch_loss: float. Training loss of current epoch.
            train: bool. Whether metrics are for training data or testing data.
        """
        if train:
            tqdm.write('-------TRAIN--------')
        else:
            tqdm.write('--------TEST--------')
        tqdm.write('--------------------')
        tqdm.write('Epoch %d, Loss %.4f' % (epoch_count, epoch_loss))
        for k, v in report.items():
            tqdm.write('%s:%.5f' % (k, v))
    
    def add_summary(
            self, epoch, train_report, train_loss,
            test_report=None, test_loss=None):
        """Outputs summary to Tensorboard writer for browser visualization.

        Args:
            epoch: int. Current epoch.
            train_report: dict. Metrics on training data.
            train_loss: float. Trainng loss.
            test_report: None|dict. Metrics on test data if present else None.
            test_loss: None|float. Loss on test data if present else None.
        """
        prefix = self.variable_name_prefix
        if test_loss:
            self.writer.add_scalars(
                '%s/loss' % prefix, {
                    'train_loss': train_loss,
                    'test_loss': test_loss},
                global_step=epoch)
        else:
            self.writer.add_scalar(
                '%s/loss' % prefix, train_loss, global_step=epoch)
        
        if test_report:
            for k in train_report:
                self.writer.add_scalars(
                    '%s/%s' % (prefix, k), {
                        ('train_%s' % k): train_report[k],
                        ('test_%s' % k): test_report[k]},
                    global_step=epoch)
        else:
            for k in train_report:
                self.writer.add_scalar(
                    '%s/%s' % (prefix, k), train_report[k],
                    global_step=epoch)
        
        train_report['loss'] = train_loss
        if test_report:
            test_report['loss'] = test_loss
            self.add_report(train_report, test_report)
        else:
            self.add_report(train_report)

    def save_report(self, name):
        """Stores the model metrics report of each epoch."""
        if not os.path.exists('./reports/%s/' % name):
            os.makedirs('./reports/%s' % name)
        f = open('./reports/%s/%s_%s.json' % (
            name, self.__class__.__name__, self.instance_time), 'w')
        json.dump(self.reports, f, indent=2)
        f.close()
